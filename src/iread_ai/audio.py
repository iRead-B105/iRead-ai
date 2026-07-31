"""업로드된 음성을 Azure Speech가 요구하는 WAV PCM으로 정규화한다.

App은 브라우저 ``MediaRecorder`` 기본값인 WebM/Opus로 녹음하고, iOS Safari에서는
MP4/M4A가 올라온다. Azure Speech SDK의 ``AudioConfig(filename=...)``은 WAV PCM만
직접 처리하므로 인식 전에 mono 16kHz 16-bit PCM으로 변환한다. 브라우저가 만든
WAV도 표본율·채널 수를 보장하지 않으므로 확장자와 무관하게 항상 변환한다.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Protocol, Sequence

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_FORMAT = "s16"


class AudioConversionError(RuntimeError):
    """음성 원본과 자격증명을 담지 않는 변환 실패 오류."""


class CompletedCommand(Protocol):
    returncode: int
    stderr: bytes | str | None


CommandRunner = Callable[..., CompletedCommand]


def convert_to_azure_wav(
    source: Path,
    *,
    ffmpeg_path: str,
    timeout_seconds: int,
    runner: CommandRunner = subprocess.run,
) -> Path:
    """``source``를 새 WAV 파일로 변환하고 그 경로를 돌려준다.

    호출자가 반환된 경로를 삭제할 책임을 진다.
    """
    with tempfile.NamedTemporaryFile(
        prefix="iread-audio-",
        suffix=".wav",
        delete=False,
    ) as temporary:
        target = Path(temporary.name)

    try:
        completed = runner(
            build_ffmpeg_command(ffmpeg_path, source, target),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exception:
        target.unlink(missing_ok=True)
        raise AudioConversionError("Audio conversion timed out") from exception
    except FileNotFoundError as exception:
        target.unlink(missing_ok=True)
        raise AudioConversionError(
            "ffmpeg is not available for audio conversion"
        ) from exception
    except OSError as exception:
        target.unlink(missing_ok=True)
        raise AudioConversionError("Audio conversion could not start") from exception

    if completed.returncode != 0:
        target.unlink(missing_ok=True)
        raise AudioConversionError("Audio conversion failed for the uploaded file")
    if not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise AudioConversionError("Audio conversion produced an empty file")
    return target


def build_ffmpeg_command(
    ffmpeg_path: str,
    source: Path,
    target: Path,
) -> Sequence[str]:
    """ffmpeg stderr에 원본 경로 외의 정보가 남지 않도록 최소 옵션만 사용한다."""
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        TARGET_SAMPLE_FORMAT,
        "-f",
        "wav",
        str(target),
    ]
