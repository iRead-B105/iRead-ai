from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    internal_api_key: str
    azure_speech_key: str
    azure_speech_region: str
    azure_speech_language: str
    max_audio_bytes: int
    audio_ffmpeg_path: str = "ffmpeg"
    audio_conversion_timeout_seconds: int = 15

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        max_audio_bytes = int(os.getenv("AI_MAX_AUDIO_BYTES", "20971520"))
        if max_audio_bytes <= 0:
            raise ValueError("AI_MAX_AUDIO_BYTES must be greater than zero")
        conversion_timeout = int(
            os.getenv("AI_AUDIO_CONVERSION_TIMEOUT_SECONDS", "15")
        )
        if conversion_timeout <= 0:
            raise ValueError(
                "AI_AUDIO_CONVERSION_TIMEOUT_SECONDS must be greater than zero"
            )
        return cls(
            internal_api_key=os.getenv("AI_INTERNAL_API_KEY", ""),
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION", ""),
            azure_speech_language=os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR"),
            max_audio_bytes=max_audio_bytes,
            audio_ffmpeg_path=os.getenv("AI_AUDIO_FFMPEG_PATH", "ffmpeg"),
            audio_conversion_timeout_seconds=conversion_timeout,
        )
