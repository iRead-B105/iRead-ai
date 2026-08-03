from datetime import timedelta
from types import SimpleNamespace

import pytest

from iread_ai.config import Settings
from iread_ai.generation_models import SpeechSynthesisRequest
from iread_ai.speech import (
    AzureSpeechProvider,
    DeterministicSpeechProvider,
    SpeechProviderError,
    _recognition_duration_ms,
    _synthesis_duration_ms,
)


class _FakeSpeechSdk:
    class ResultReason:
        SynthesizingAudioCompleted = "completed"

    class SpeechSynthesisOutputFormat:
        Audio16Khz32KBitRateMonoMp3 = "mp3"

    def __init__(self) -> None:
        owner = self

        class SpeechConfig:
            def __init__(self, **_: str) -> None:
                self.speech_recognition_language = ""
                self.speech_synthesis_voice_name = ""

            def set_speech_synthesis_output_format(self, _: str) -> None:
                pass

        class SpeechSynthesizer:
            def __init__(self, *, speech_config, audio_config) -> None:
                del speech_config
                assert audio_config is None

            def speak_text_async(self, text: str):
                assert text == "테스트 음성"
                result = SimpleNamespace(
                    reason=owner.ResultReason.SynthesizingAudioCompleted,
                    audio_duration=123,
                    audio_data=b"ID3-result-audio",
                )
                return SimpleNamespace(get=lambda: result)

        self.SpeechConfig = SpeechConfig
        self.SpeechSynthesizer = SpeechSynthesizer


def test_azure_recognition_ticks_are_converted_to_milliseconds() -> None:
    assert _recognition_duration_ms(12_340_000) == 1234


def test_duration_helpers_accept_timedelta_results() -> None:
    duration = timedelta(milliseconds=1234)

    assert _recognition_duration_ms(duration) == 1234
    assert _synthesis_duration_ms(duration) == 1234


def test_local_transcription_never_echoes_expected_text() -> None:
    result = DeterministicSpeechProvider().transcribe(
        request_id="speech-1",
        audio=b"browser-recording",
        original_filename="recording.webm",
    )

    assert result.transcript == ""
    assert result.confidence == 0
    assert result.durationMs == 0


def test_local_synthesis_fails_instead_of_returning_silent_mp3() -> None:
    request = SpeechSynthesisRequest(
        requestId="tts-1",
        text="가",
    )

    with pytest.raises(SpeechProviderError, match="AI_SPEECH_PROVIDER=azure"):
        DeterministicSpeechProvider().synthesize(request)


def test_azure_synthesis_returns_sdk_audio_without_a_temporary_file(monkeypatch) -> None:
    fake_sdk = _FakeSpeechSdk()
    provider = AzureSpeechProvider(
        Settings(
            _env_file=None,
            internal_api_key="test-key",
            azure_speech_key="azure-key",
            azure_speech_region="koreacentral",
        )
    )
    monkeypatch.setattr(provider, "_sdk", lambda: fake_sdk)

    result = provider.synthesize(
        SpeechSynthesisRequest(requestId="tts-temp-cleanup", text="테스트 음성")
    )

    assert result.audio == b"ID3-result-audio"
    assert result.duration_ms == 123
