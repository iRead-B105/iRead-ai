from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import pytest

from iread_ai.audio import (
    AudioConversionError,
    build_ffmpeg_command,
    convert_to_azure_wav,
)


@dataclass
class StubCompleted:
    returncode: int
    stderr: bytes | None = None


def _runner(returncode: int, *, write_output: bool = True):
    calls: list[list[str]] = []

    def run(command: list[str], **_: Any) -> StubCompleted:
        calls.append(list(command))
        if write_output:
            Path(command[-1]).write_bytes(b"RIFF....WAVEfmt ")
        return StubCompleted(returncode=returncode)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_builds_mono_16khz_pcm_command(tmp_path: Path) -> None:
    command = build_ffmpeg_command(
        "ffmpeg",
        tmp_path / "input.webm",
        tmp_path / "output.wav",
    )

    assert command[0] == "ffmpeg"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-sample_fmt") + 1] == "s16"
    assert command[command.index("-f") + 1] == "wav"
    assert command[-1] == str(tmp_path / "output.wav")


def test_converts_webm_upload_to_wav(tmp_path: Path) -> None:
    source = tmp_path / "recording.webm"
    source.write_bytes(b"webm payload")
    run = _runner(0)

    target = convert_to_azure_wav(
        source,
        ffmpeg_path="ffmpeg",
        timeout_seconds=15,
        runner=run,
    )

    try:
        assert target.suffix == ".wav"
        assert target.exists()
        assert run.calls[0][run.calls[0].index("-i") + 1] == str(source)
    finally:
        target.unlink(missing_ok=True)


def test_reports_failure_without_leaking_the_target_file(tmp_path: Path) -> None:
    source = tmp_path / "broken.webm"
    source.write_bytes(b"not audio")
    run = _runner(1)

    with pytest.raises(AudioConversionError):
        convert_to_azure_wav(
            source,
            ffmpeg_path="ffmpeg",
            timeout_seconds=15,
            runner=run,
        )

    assert not Path(run.calls[0][-1]).exists()


def test_reports_empty_output_as_failure(tmp_path: Path) -> None:
    source = tmp_path / "silent.webm"
    source.write_bytes(b"silence")
    run = _runner(0, write_output=False)

    with pytest.raises(AudioConversionError):
        convert_to_azure_wav(
            source,
            ffmpeg_path="ffmpeg",
            timeout_seconds=15,
            runner=run,
        )


def test_reports_timeout_as_failure(tmp_path: Path) -> None:
    source = tmp_path / "long.webm"
    source.write_bytes(b"long audio")
    captured: list[str] = []

    def run(command: list[str], **_: Any) -> StubCompleted:
        captured.append(command[-1])
        raise subprocess.TimeoutExpired(cmd=command, timeout=15)

    with pytest.raises(AudioConversionError, match="timed out"):
        convert_to_azure_wav(
            source,
            ffmpeg_path="ffmpeg",
            timeout_seconds=15,
            runner=run,
        )

    assert not Path(captured[0]).exists()


def test_reports_missing_ffmpeg_binary(tmp_path: Path) -> None:
    source = tmp_path / "recording.webm"
    source.write_bytes(b"webm payload")

    with pytest.raises(AudioConversionError, match="not available"):
        convert_to_azure_wav(
            source,
            ffmpeg_path="iread-ffmpeg-does-not-exist",
            timeout_seconds=15,
        )
