from __future__ import annotations

import json
import mimetypes
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from .audio import AudioPreparationError, stage_azure_audio
from .config import Settings
from .models import PronunciationAnalysisResponse, PronunciationWordResult

ANALYSIS_VERSION = "AZURE_SPEECH_KO_KR_WORD_V1"
DETERMINISTIC_ANALYSIS_VERSION = "IREAD_DETERMINISTIC_KO_KR_WORD_V1"
GMS_STT_ANALYSIS_VERSION = "GMS_WHISPER1_STT_READING_MATCH_KO_KR_V1"
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

        path: Path | None = None
        try:
            path = stage_azure_audio(audio, original_filename)
            payload = self._recognize(reference_text, path)
            return parse_azure_result(request_id=request_id, payload=payload)
        except AudioPreparationError as exception:
            raise PronunciationProviderError(str(exception)) from exception
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


class GMSPronunciationProvider:
    """Audio-dependent reading score backed by GMS Whisper transcription.

    This provider measures how closely recognized Korean text matches the
    scripted reference. It is useful when Azure Pronunciation Assessment is
    unavailable, but it does not claim phoneme- or prosody-level assessment.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def analyze(
        self,
        *,
        request_id: str,
        reference_text: str,
        audio: bytes,
        original_filename: str | None,
    ) -> PronunciationAnalysisResponse:
        if self._settings.gms_key is None:
            raise PronunciationProviderError("GMS key is not configured")
        api_key = self._settings.gms_key.get_secret_value()
        if not api_key:
            raise PronunciationProviderError("GMS key is not configured")

        filename = Path(original_filename or "recording.wav").name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        try:
            with httpx.Client(
                timeout=self._settings.gms_speech_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = client.post(
                    f"{self._settings.gms_text_base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={
                        "model": self._settings.gms_speech_model,
                        "language": "ko",
                        "response_format": "json",
                    },
                    files={"file": (filename, audio, content_type)},
                )
        except httpx.HTTPError as exception:
            raise PronunciationProviderError(
                "GMS speech transcription could not be reached"
            ) from exception

        if not response.is_success:
            raise PronunciationProviderError(
                f"GMS speech transcription failed with status {response.status_code}"
            )
        try:
            recognized_text = str(response.json().get("text", "")).strip()
        except (AttributeError, TypeError, ValueError) as exception:
            raise PronunciationProviderError(
                "GMS speech transcription returned an invalid result"
            ) from exception

        return score_transcribed_reading(
            request_id=request_id,
            reference_text=reference_text,
            recognized_text=recognized_text,
        )


class DeterministicPronunciationProvider:
    """Local integration provider used when Azure credentials are unavailable.

    It keeps the Backend -> AI contract real without pretending to be a
    production speech assessment. Production must select the Azure provider.
    """

    def analyze(
        self,
        *,
        request_id: str,
        reference_text: str,
        audio: bytes,
        original_filename: str | None,
    ) -> PronunciationAnalysisResponse:
        del audio, original_filename
        words = reference_text.split() or [reference_text]
        return PronunciationAnalysisResponse(
            requestId=request_id,
            pronunciationAccuracyScore=0.0,
            fluencyScore=0.0,
            completenessScore=0.0,
            pronScore=0.0,
            confidence=0.0,
            analysisVersion=DETERMINISTIC_ANALYSIS_VERSION,
            recognizedText=None,
            words=[
                PronunciationWordResult(
                    resultIndex=index,
                    word=word,
                    accuracyScore=0.0,
                    errorType="Omission",
                    offsetMs=0,
                    durationMs=0,
                )
                for index, word in enumerate(words)
            ],
        )


def score_transcribed_reading(
    *,
    request_id: str,
    reference_text: str,
    recognized_text: str,
) -> PronunciationAnalysisResponse:
    expected_units = _korean_units(reference_text)
    recognized_units = _korean_units(recognized_text)
    if not expected_units:
        raise PronunciationProviderError("Reference text has no scorable characters")

    distance = _edit_distance(expected_units, recognized_units)
    accuracy = _bounded_score(
        100 * (1 - distance / max(len(expected_units), len(recognized_units), 1))
    )
    matched_units = sum(
        block.size
        for block in SequenceMatcher(
            None,
            expected_units,
            recognized_units,
            autojunk=False,
        ).get_matching_blocks()
    )
    completeness = _bounded_score(100 * matched_units / len(expected_units))

    return PronunciationAnalysisResponse(
        requestId=request_id,
        pronunciationAccuracyScore=accuracy,
        fluencyScore=None,
        completenessScore=completeness,
        pronScore=accuracy,
        confidence=0.0,
        analysisVersion=GMS_STT_ANALYSIS_VERSION,
        recognizedText=recognized_text,
        words=_align_transcribed_words(reference_text, recognized_text),
    )


def _align_transcribed_words(
    reference_text: str,
    recognized_text: str,
) -> list[PronunciationWordResult]:
    expected_words = reference_text.split() or [reference_text]
    recognized_words = recognized_text.split()
    matcher = SequenceMatcher(None, expected_words, recognized_words, autojunk=False)
    results: list[PronunciationWordResult] = []

    def append(word: str, score: float | None, error_type: str) -> None:
        results.append(
            PronunciationWordResult(
                resultIndex=len(results),
                word=word,
                accuracyScore=score,
                errorType=error_type,
                offsetMs=0,
                durationMs=0,
            )
        )

    for tag, expected_start, expected_end, actual_start, actual_end in matcher.get_opcodes():
        expected = expected_words[expected_start:expected_end]
        actual = recognized_words[actual_start:actual_end]
        if tag == "equal":
            for word in expected:
                append(word, 100.0, "None")
            continue
        if tag == "delete":
            for word in expected:
                append(word, 0.0, "Omission")
            continue
        if tag == "insert":
            for word in actual:
                append(word, 0.0, "Insertion")
            continue

        paired = min(len(expected), len(actual))
        for index in range(paired):
            expected_word = expected[index]
            expected_word_units = _korean_units(expected_word)
            actual_word_units = _korean_units(actual[index])
            distance = _edit_distance(expected_word_units, actual_word_units)
            score = _bounded_score(
                100
                * (
                    1
                    - distance
                    / max(len(expected_word_units), len(actual_word_units), 1)
                )
            )
            append(expected_word, score, "None" if score >= 90 else "Mispronunciation")
        for word in expected[paired:]:
            append(word, 0.0, "Omission")
        for word in actual[paired:]:
            append(word, 0.0, "Insertion")

    if not results:
        for word in expected_words:
            append(word, 0.0, "Omission")
    return results


def _korean_units(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", value.casefold())
    return [character for character in normalized if character.isalnum()]


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _bounded_score(value: float) -> float:
    return round(min(100.0, max(0.0, value)), 2)


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
        recognizedText=_optional_text(
            best.get("Display") or payload.get("DisplayText")
        ),
        words=words,
    )


def _ticks_to_ms(value: object) -> int:
    return max(0, int(value) // TICKS_PER_MILLISECOND)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
