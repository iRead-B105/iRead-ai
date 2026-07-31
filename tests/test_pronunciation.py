from pathlib import Path
from typing import Any

import pytest

from iread_ai.config import Settings
from iread_ai.pronunciation import (
    AzurePronunciationProvider,
    PronunciationProviderError,
    parse_azure_result,
)


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


def test_removes_uploaded_and_converted_audio_after_analysis() -> None:
    class StubProvider(AzurePronunciationProvider):
        uploaded_path: Path | None = None
        converted_path: Path | None = None

        def _to_recognizable_wav(self, uploaded: Path) -> Path:
            self.uploaded_path = uploaded
            assert uploaded.exists()
            converted = uploaded.with_suffix(".converted.wav")
            converted.write_bytes(b"RIFF....WAVEfmt ")
            self.converted_path = converted
            return converted

        def _recognize(
            self,
            reference_text: str,
            audio_path: Path,
        ) -> dict[str, Any]:
            assert audio_path == self.converted_path
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
        )
    )

    provider.analyze(
        request_id="request-3",
        reference_text="사과",
        audio=b"temporary audio",
        original_filename="word.webm",
    )

    assert provider.uploaded_path is not None
    assert provider.converted_path is not None
    assert not provider.uploaded_path.exists()
    assert not provider.converted_path.exists()


def test_conversion_failure_becomes_a_safe_upstream_error() -> None:
    provider = AzurePronunciationProvider(
        Settings(
            internal_api_key="internal",
            azure_speech_key="azure-key",
            azure_speech_region="koreacentral",
            azure_speech_language="ko-KR",
            max_audio_bytes=1024,
            audio_ffmpeg_path="iread-ffmpeg-does-not-exist",
        )
    )

    with pytest.raises(PronunciationProviderError, match="ffmpeg"):
        provider.analyze(
            request_id="request-4",
            reference_text="사과",
            audio=b"webm payload",
            original_filename="word.webm",
        )
