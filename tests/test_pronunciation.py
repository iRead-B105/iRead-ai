from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from iread_ai.config import Settings
from iread_ai.pronunciation import (
    AzurePronunciationProvider,
    DeterministicPronunciationProvider,
    GMSPronunciationProvider,
    PronunciationProviderError,
    parse_azure_result,
    score_transcribed_reading,
    unrecognized_pronunciation_result,
)


def _azure_settings() -> Settings:
    return Settings(
        internal_api_key="internal",
        azure_speech_key="azure-key",
        azure_speech_region="koreacentral",
        azure_speech_language="ko-KR",
        max_audio_bytes=1024,
        pronunciation_provider="azure",
    )


def _stub_azure_sdk(monkeypatch: pytest.MonkeyPatch, reason_name: str) -> Any:
    """실제 SDK 의 ResultReason 을 그대로 쓰고 인식기만 대체한다."""
    speechsdk = pytest.importorskip("azure.cognitiveservices.speech")

    class StubRecognizer:
        def __init__(self, **_: Any) -> None:
            pass

        def recognize_once_async(self) -> Any:
            reason = getattr(speechsdk.ResultReason, reason_name)
            return SimpleNamespace(get=lambda: SimpleNamespace(reason=reason))

    monkeypatch.setattr(
        speechsdk,
        "SpeechConfig",
        lambda **_: SimpleNamespace(speech_recognition_language=None),
    )
    monkeypatch.setattr(speechsdk.audio, "AudioConfig", lambda **_: object())
    monkeypatch.setattr(
        speechsdk,
        "PronunciationAssessmentConfig",
        lambda **_: SimpleNamespace(apply_to=lambda _recognizer: None),
    )
    monkeypatch.setattr(speechsdk, "SpeechRecognizer", StubRecognizer)
    return speechsdk


def test_local_pronunciation_never_awards_points_from_audio_length() -> None:
    result = DeterministicPronunciationProvider().analyze(
        request_id="local-1",
        reference_text="가 나",
        audio=b"not-real-speech" * 10_000,
        original_filename="recording.webm",
    )

    assert result.pronunciationAccuracyScore == 0
    assert result.confidence == 0
    assert [word.errorType for word in result.words] == ["Omission", "Omission"]


def test_parses_sentence_into_word_scores_and_milliseconds() -> None:
    result = parse_azure_result(
        request_id="request-1",
        payload={
            "NBest": [
                {
                    "Confidence": 0.94,
                    "PronunciationAssessment": {
                        "AccuracyScore": 82,
                        "FluencyScore": 79,
                        "CompletenessScore": 66,
                        "PronScore": 78,
                    },
                    "Words": [
                        {
                            "Word": "아기는",
                            "Offset": 1_000_000,
                            "Duration": 5_000_000,
                            "PronunciationAssessment": {
                                "AccuracyScore": 91,
                                "ErrorType": "None",
                            },
                        },
                        {
                            "Word": "사과를",
                            "Offset": 0,
                            "Duration": 0,
                            "PronunciationAssessment": {
                                "ErrorType": "Omission",
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert result.pronunciationAccuracyScore == 82
    assert result.words[0].offsetMs == 100
    assert result.words[0].durationMs == 500
    assert result.words[1].accuracyScore is None
    assert result.words[1].errorType == "Omission"


def test_keeps_insertions_for_backend_alignment() -> None:
    result = parse_azure_result(
        request_id="request-2",
        payload={
            "NBest": [
                {
                    "Confidence": 0.9,
                    "PronunciationAssessment": {"AccuracyScore": 90},
                    "Words": [
                        {
                            "Word": "정말",
                            "Offset": 1_000_000,
                            "Duration": 2_000_000,
                            "PronunciationAssessment": {
                                "AccuracyScore": 80,
                                "ErrorType": "Insertion",
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert result.words[0].resultIndex == 0
    assert result.words[0].errorType == "Insertion"


def test_scores_perfect_gms_transcription_as_full_match() -> None:
    result = score_transcribed_reading(
        request_id="gms-perfect",
        reference_text="토끼가 숲길을 걸어요.",
        recognized_text="토끼가 숲길을 걸어요",
    )

    assert result.pronunciationAccuracyScore == 100
    assert result.completenessScore == 100
    assert result.fluencyScore is None
    assert result.recognizedText == "토끼가 숲길을 걸어요"
    assert [word.errorType for word in result.words] == ["None", "None", "None"]


def test_scores_gms_transcription_from_actual_audio_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/audio/transcriptions")
        assert request.headers["Authorization"] == "Bearer gms-key"
        assert b'recording.wav' in request.content
        return httpx.Response(200, json={"text": "토끼가 숲길을 걸어요"})

    provider = GMSPronunciationProvider(
        Settings(
            internal_api_key="internal",
            pronunciation_provider="gms",
            gms_key="gms-key",
            gms_openai_base_url="https://example.test/v1",
        ),
        transport=httpx.MockTransport(handler),
    )

    result = provider.analyze(
        request_id="gms-audio",
        reference_text="토끼가 숲길을 걸어요.",
        audio=b"real-audio-payload",
        original_filename="recording.wav",
    )

    assert result.pronunciationAccuracyScore == 100
    assert result.analysisVersion.startswith("GMS_WHISPER1")
    assert result.recognizedText == "토끼가 숲길을 걸어요"


def test_gms_empty_transcription_is_scored_as_omission() -> None:
    result = score_transcribed_reading(
        request_id="gms-empty",
        reference_text="사과를 먹어요",
        recognized_text="",
    )

    assert result.pronunciationAccuracyScore == 0
    assert result.completenessScore == 0
    assert [word.errorType for word in result.words] == ["Omission", "Omission"]


def test_removes_temporary_audio_after_analysis() -> None:
    class StubProvider(AzurePronunciationProvider):
        staged_path: Path | None = None

        def _recognize(
            self,
            reference_text: str,
            audio_path: Path,
        ) -> dict[str, Any]:
            self.staged_path = audio_path
            assert audio_path.exists()
            return {
                "NBest": [
                    {
                        "Confidence": 0.9,
                        "PronunciationAssessment": {"AccuracyScore": 90},
                        "Words": [
                            {
                                "Word": reference_text,
                                "Offset": 0,
                                "Duration": 1_000_000,
                                "PronunciationAssessment": {
                                    "AccuracyScore": 90,
                                    "ErrorType": "None",
                                },
                            }
                        ],
                    }
                ]
            }

    provider = StubProvider(
        Settings(
            internal_api_key="internal",
            azure_speech_key="azure-key",
            azure_speech_region="koreacentral",
            azure_speech_language="ko-KR",
            max_audio_bytes=1024,
            pronunciation_provider="azure",
        )
    )

    provider.analyze(
        request_id="request-3",
        reference_text="사과",
        audio=b"temporary audio",
        original_filename="word.wav",
    )

    assert provider.staged_path is not None
    assert not provider.staged_path.exists()


def test_azure_no_match_is_scored_as_zero_instead_of_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """아이가 알아듣기 어렵게 읽으면 Azure 는 NoMatch 를 준다.

    낱말 읽기에서 이 경우 502 가 나서 화면에 'AI 처리 중 오류가 발생했습니다'가
    떴다. 평가 실패가 아니라 0점이므로 정상 응답이어야 한다.
    """
    _stub_azure_sdk(monkeypatch, "NoMatch")

    result = AzurePronunciationProvider(_azure_settings()).analyze(
        request_id="no-match-1",
        reference_text="사과",
        audio=b"unintelligible audio",
        original_filename="word.wav",
    )

    assert result.pronunciationAccuracyScore == 0
    assert result.pronScore == 0
    assert result.confidence == 0
    assert result.recognizedText is None
    assert [word.word for word in result.words] == ["사과"]
    assert [word.errorType for word in result.words] == ["Omission"]
    # Azure 가 평가한 결과이므로 분석 버전은 Azure 로 남는다.
    assert result.analysisVersion == "AZURE_SPEECH_KO_KR_WORD_V1"


def test_azure_cancellation_is_still_an_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """인증·요금·네트워크 실패(Canceled)는 계속 업스트림 오류로 올린다."""
    _stub_azure_sdk(monkeypatch, "Canceled")

    with pytest.raises(PronunciationProviderError) as error:
        AzurePronunciationProvider(_azure_settings()).analyze(
            request_id="canceled-1",
            reference_text="사과",
            audio=b"audio",
            original_filename="word.wav",
        )

    assert "Canceled" in str(error.value)
    # 자격증명이 섞일 수 있는 error_details 는 노출하지 않는다.
    assert "azure-key" not in str(error.value)


def test_unrecognized_result_splits_words_like_backend_reference_rule() -> None:
    """Backend 는 [가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+ 로 기준 단어를 자른다.

    개수나 표기가 어긋나면 Backend 가 정렬 실패(409)를 내므로 같은 규칙을 쓴다.
    """
    result = unrecognized_pronunciation_result(
        request_id="no-match-2",
        reference_text="사과를 먹어요.",
    )

    assert [word.word for word in result.words] == ["사과를", "먹어요"]
    assert all(word.accuracyScore == 0 for word in result.words)
