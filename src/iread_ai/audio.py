from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile


class AudioPreparationError(RuntimeError):
    """Raised when uploaded browser audio cannot be normalized for Azure."""


_SUPPORTED_SUFFIXES = {".wav", ".webm", ".mp3", ".mp4", ".m4a", ".ogg"}


def stage_azure_audio(audio: bytes, original_filename: str | None) -> Path:
    """Stage audio as a 16 kHz mono PCM WAV accepted by Azure Speech.

    Browser MediaRecorder normally emits WebM/Opus (or MP4/AAC on Safari),
    while Azure Speech's filename input expects a WAV container. WAV uploads
    are kept as-is; compressed browser recordings are normalized with ffmpeg.
    The caller owns and must delete the returned path.
    """

    suffix = Path(original_filename or "recording.audio").suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        suffix = ".audio"

    with tempfile.NamedTemporaryFile(
        prefix="iread-audio-source-",
        suffix=suffix,
        delete=False,
    ) as temporary:
        temporary.write(audio)
        source_path = Path(temporary.name)

    if suffix == ".wav":
        return source_path

    with tempfile.NamedTemporaryFile(
        prefix="iread-audio-azure-",
        suffix=".wav",
        delete=False,
    ) as temporary:
        output_path = Path(temporary.name)

    prepared = False
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            timeout=20,
        )
        if completed.returncode != 0 or output_path.stat().st_size == 0:
            raise AudioPreparationError(
                "Uploaded audio format could not be decoded"
            )
        prepared = True
        return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exception:
        raise AudioPreparationError(
            "Uploaded audio could not be prepared for speech recognition"
        ) from exception
    finally:
        source_path.unlink(missing_ok=True)
        if not prepared:
            output_path.unlink(missing_ok=True)
