from datetime import timedelta

import pytest

from iread_ai.generation_models import SpeechSynthesisRequest
from iread_ai.speech import (
    DeterministicSpeechProvider,
    SpeechProviderError,
    _recognition_duration_ms,
    _synthesis_duration_ms,
)


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
