from pathlib import Path
import subprocess

import pytest

from iread_ai.audio import AudioPreparationError, stage_azure_audio


def test_keeps_wav_upload_without_invoking_ffmpeg(monkeypatch) -> None:
    def unexpected_run(*args, **kwargs):
        raise AssertionError("ffmpeg must not run for WAV uploads")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    path = stage_azure_audio(b"RIFF-test-wave", "voice.wav")
    try:
        assert path.suffix == ".wav"
        assert path.read_bytes() == b"RIFF-test-wave"
    finally:
        path.unlink(missing_ok=True)


def test_converts_webm_upload_to_pcm_wav(monkeypatch) -> None:
    source_path: Path | None = None

    def successful_run(command, **kwargs):
        nonlocal source_path
        source_path = Path(command[command.index("-i") + 1])
        output_path = Path(command[-1])
        assert source_path.read_bytes() == b"webm-opus"
        assert command[command.index("-ac") + 1] == "1"
        assert command[command.index("-ar") + 1] == "16000"
        assert command[command.index("-c:a") + 1] == "pcm_s16le"
        output_path.write_bytes(b"RIFF-normalized-wave")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", successful_run)
    path = stage_azure_audio(b"webm-opus", "voice.webm")
    try:
        assert path.suffix == ".wav"
        assert path.read_bytes() == b"RIFF-normalized-wave"
        assert source_path is not None
        assert not source_path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_rejects_audio_ffmpeg_cannot_decode(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )

    with pytest.raises(AudioPreparationError, match="could not be decoded"):
        stage_azure_audio(b"invalid", "voice.webm")
