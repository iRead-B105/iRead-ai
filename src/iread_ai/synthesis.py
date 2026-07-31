"""Azure Speech로 한국어 문장을 읽어 주는 음성을 만든다.

Backend는 이야기 낭독과 따라 읽기 예시 음성에 이 결과를 쓴다. 발음 평가와 같은
자격증명을 쓰지만 오류 메시지에는 키와 원문을 담지 않는다.
"""

from __future__ import annotations

from .config import Settings

SYNTHESIS_VERSION = "AZURE_SPEECH_KO_KR_TTS_V1"
DEFAULT_VOICE = "ko-KR-SunHiNeural"
TICKS_PER_MILLISECOND = 10_000


class SpeechSynthesisError(RuntimeError):
    """자격증명과 원문을 담지 않는 안전한 상위 오류."""


class AzureSpeechSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def synthesize(self, *, text: str, voice: str | None) -> tuple[bytes, int]:
        """mp3 바이트와 재생 시간(ms)을 돌려준다."""
        if not self._settings.azure_speech_key:
            raise SpeechSynthesisError("Azure Speech key is not configured")
        if not self._settings.azure_speech_region:
            raise SpeechSynthesisError("Azure Speech region is not configured")

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exception:
            raise SpeechSynthesisError(
                "Azure Speech SDK is not installed"
            ) from exception

        config = speechsdk.SpeechConfig(
            subscription=self._settings.azure_speech_key,
            region=self._settings.azure_speech_region,
        )
        config.speech_synthesis_voice_name = voice or DEFAULT_VOICE
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        # audio_config=None이면 파일로 쓰지 않고 결과 바이트만 받는다.
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            raise SpeechSynthesisError("Azure Speech did not synthesize the text")
        audio = bytes(result.audio_data)
        if not audio:
            raise SpeechSynthesisError("Azure Speech returned empty audio")
        return audio, duration_ms(result)


def duration_ms(result: object) -> int:
    """SDK 버전에 따라 timedelta 또는 100ns tick으로 재생 시간을 준다."""
    duration = getattr(result, "audio_duration", None)
    if duration is None:
        return 0
    total_seconds = getattr(duration, "total_seconds", None)
    if callable(total_seconds):
        return max(0, int(total_seconds() * 1000))
    return max(0, int(duration) // TICKS_PER_MILLISECOND)
