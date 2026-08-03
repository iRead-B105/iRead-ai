from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .audio import AudioPreparationError, stage_azure_audio
from .config import Settings
from .generation_models import (
    SpeechSynthesisRequest,
    SpeechTranscriptionResponse,
)


class SpeechProviderError(RuntimeError):
    """Safe speech provider failure without credentials or raw audio."""


def _recognition_duration_ms(duration: object) -> int:
    """Convert Azure recognition duration to milliseconds.

    Current Azure Speech SDK versions expose recognition duration as integer
    ticks (100 ns each). Older result doubles used by tests and integrations may
    expose a timedelta instead, so support both representations.
    """
    total_seconds = getattr(duration, "total_seconds", None)
    if callable(total_seconds):
        return max(0, int(total_seconds() * 1000))
    return max(0, int(duration) // 10_000)


def _synthesis_duration_ms(duration: object) -> int:
    """Convert Azure synthesis duration to milliseconds."""
    total_seconds = getattr(duration, "total_seconds", None)
    if callable(total_seconds):
        return max(0, int(total_seconds() * 1000))
    return max(0, int(duration))


@dataclass(frozen=True)
class SynthesizedSpeech:
    audio: bytes
    media_type: str
    duration_ms: int


class AzureSpeechProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def transcribe(
        self,
        *,
        request_id: str,
        audio: bytes,
        original_filename: str | None,
    ) -> SpeechTranscriptionResponse:
        speechsdk = self._sdk()
        try:
            path = stage_azure_audio(audio, original_filename)
        except AudioPreparationError as exception:
            raise SpeechProviderError(str(exception)) from exception
        try:
            speech_config = self._speech_config(speechsdk)
            audio_config = speechsdk.audio.AudioConfig(filename=str(path))
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.NoMatch:
                return SpeechTranscriptionResponse(
                    requestId=request_id,
                    transcript="",
                    confidence=0,
                    durationMs=0,
                )
            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                raise SpeechProviderError("Azure Speech recognition failed")
            duration_ms = _recognition_duration_ms(result.duration)
            return SpeechTranscriptionResponse(
                requestId=request_id,
                transcript=(result.text or "").strip(),
                confidence=1.0 if result.text else 0.0,
                durationMs=duration_ms,
            )
        finally:
            path.unlink(missing_ok=True)

    def synthesize(self, request: SpeechSynthesisRequest) -> SynthesizedSpeech:
        speechsdk = self._sdk()
        speech_config = self._speech_config(speechsdk)
        voice = (request.voice or self._settings.azure_speech_voice).strip()
        if voice:
            speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        output_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="iread-tts-", suffix=".mp3", delete=False
            ) as temporary:
                output_path = Path(temporary.name)
            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=str(output_path)
            )
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(request.text).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                raise SpeechProviderError("Azure Speech synthesis failed")
            audio = output_path.read_bytes()
            if not audio:
                raise SpeechProviderError("Azure Speech returned empty audio")
            return SynthesizedSpeech(
                audio=audio,
                media_type="audio/mpeg",
                duration_ms=_synthesis_duration_ms(result.audio_duration),
            )
        finally:
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    def _speech_config(self, speechsdk):
        if not self._settings.azure_speech_key:
            raise SpeechProviderError("Azure Speech key is not configured")
        if not self._settings.azure_speech_region:
            raise SpeechProviderError("Azure Speech region is not configured")
        config = speechsdk.SpeechConfig(
            subscription=self._settings.azure_speech_key,
            region=self._settings.azure_speech_region,
        )
        config.speech_recognition_language = self._settings.azure_speech_language
        return config

    @staticmethod
    def _sdk():
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exception:
            raise SpeechProviderError("Azure Speech SDK is not installed") from exception
        return speechsdk

class DeterministicSpeechProvider:
    """Honest local fallback.

    It intentionally never derives a transcript from expectedText. Local mode
    can exercise the HTTP/UI failure path, but only a real provider may report
    recognized speech or return synthesized child speech.
    """

    def transcribe(
        self,
        *,
        request_id: str,
        audio: bytes,
        original_filename: str | None,
    ) -> SpeechTranscriptionResponse:
        del audio, original_filename
        return SpeechTranscriptionResponse(
            requestId=request_id,
            transcript="",
            confidence=0,
            durationMs=0,
        )

    def synthesize(self, request: SpeechSynthesisRequest) -> SynthesizedSpeech:
        del request
        raise SpeechProviderError(
            "Speech synthesis requires AI_SPEECH_PROVIDER=azure"
        )
