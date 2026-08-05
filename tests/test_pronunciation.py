from pathlib import Path
from typing import Any

import httpx

from iread_ai.config import Settings
from iread_ai.pronunciation import (
    AzurePronunciationProvider,
    DeterministicPronunciationProvider,
    GMSPronunciationProvider,
    parse_azure_result,
    score_transcribed_reading,
)


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
