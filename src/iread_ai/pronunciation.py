from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from .config import Settings
from .models import PronunciationAnalysisResponse, PronunciationWordResult


ANALYSIS_VERSION = "AZURE_SPEECH_KO_KR_WORD_V1"
TICKS_PER_MILLISECOND = 10_000


class PronunciationProviderError(RuntimeError):
    """Safe upstream error that never contains credentials or raw audio."""


class AzurePronunciationProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def analyze(
        self,
        *,
        request_id: str,
        reference_text: str,
        audio: bytes,
        original_filename: str | None,
    ) -> PronunciationAnalysisResponse:
        if not self._settings.azure_speech_key:
            raise PronunciationProviderError("Azure Speech key is not configured")
        if not self._settings.azure_speech_region:
            raise PronunciationProviderError("Azure Speech region is not configured")

        suffix = _safe_suffix(original_filename)
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="iread-pronunciation-",
                suffix=suffix,
                delete=False,
            ) as temporary:
                temporary.write(audio)
                path = Path(temporary.name)
            payload = self._recognize(reference_text, path)
            return parse_azure_result(request_id=request_id, payload=payload)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _recognize(self, reference_text: str, audio_path: Path) -> dict[str, Any]:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exception:
            raise PronunciationProviderError(
                "Azure Speech SDK is not installed"
            ) from exception

        speech_config = speechsdk.SpeechConfig(
            subscription=self._settings.azure_speech_key,
            region=self._settings.azure_speech_region,
        )
        speech_config.speech_recognition_language = (
            self._settings.azure_speech_language
        )
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        assessment = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word,
            enable_miscue=True,
        )
        assessment.apply_to(recognizer)
        result = recognizer.recognize_once_async().get()
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            raise PronunciationProviderError("Azure Speech did not recognize the audio")
        raw = result.properties.get(
            speechsdk.PropertyId.SpeechServiceResponse_JsonResult
        )
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exception:
            raise PronunciationProviderError(
                "Azure Speech returned an invalid result"
            ) from exception


def parse_azure_result(
    *,
    request_id: str,
    payload: dict[str, Any],
) -> PronunciationAnalysisResponse:
    try:
        best = payload["NBest"][0]
        assessment = best["PronunciationAssessment"]
        raw_words = best["Words"]
    except (KeyError, IndexError, TypeError) as exception:
        raise PronunciationProviderError(
            "Azure Speech result is missing pronunciation fields"
        ) from exception

    words: list[PronunciationWordResult] = []
    for index, raw_word in enumerate(raw_words):
        word_assessment = raw_word.get("PronunciationAssessment", {})
        words.append(
            PronunciationWordResult(
                resultIndex=index,
                word=str(raw_word["Word"]),
                accuracyScore=_optional_float(
                    word_assessment.get("AccuracyScore")
                ),
                errorType=str(word_assessment.get("ErrorType", "None")),
                offsetMs=_ticks_to_ms(raw_word.get("Offset", 0)),
                durationMs=_ticks_to_ms(raw_word.get("Duration", 0)),
            )
        )

    return PronunciationAnalysisResponse(
        requestId=request_id,
        pronunciationAccuracyScore=float(assessment["AccuracyScore"]),
        fluencyScore=_optional_float(assessment.get("FluencyScore")),
        completenessScore=_optional_float(
            assessment.get("CompletenessScore")
        ),
        pronScore=_optional_float(assessment.get("PronScore")),
        confidence=float(best.get("Confidence", 0)),
        analysisVersion=ANALYSIS_VERSION,
        words=words,
    )


def _ticks_to_ms(value: object) -> int:
    return max(0, int(value) // TICKS_PER_MILLISECOND)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".audio"
    suffix = Path(filename).suffix.lower()
    if suffix in {".wav", ".webm", ".mp3", ".mp4", ".m4a"}:
        return suffix
    return ".audio"
