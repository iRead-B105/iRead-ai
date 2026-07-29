from pathlib import Path
from typing import Any

from iread_ai.config import Settings
from iread_ai.pronunciation import AzurePronunciationProvider, parse_azure_result


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
