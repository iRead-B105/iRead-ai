from __future__ import annotations

from dataclasses import dataclass
import math
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    internal_api_key: str
    azure_speech_key: str
    azure_speech_region: str
    azure_speech_language: str
    max_audio_bytes: int
    app_env: str = "development"
    generation_provider: str = "mock"
    gms_key: str = ""
    gms_base_url: str = "https://gms.ssafy.io/gmsapi"
    gms_openai_base_url: str = ""
    gms_text_model: str = "gpt-5.4-mini"
    gms_text_timeout_seconds: float = 28.0
    gms_max_output_tokens: int = 3200
    idempotency_ttl_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        max_audio_bytes = int(os.getenv("AI_MAX_AUDIO_BYTES", "20971520"))
        if max_audio_bytes <= 0:
            raise ValueError("AI_MAX_AUDIO_BYTES must be greater than zero")
        provider = os.getenv(
            "AI_GENERATION_PROVIDER", os.getenv("STORY_PROVIDER", "mock")
        ).lower()
        if provider not in {"mock", "gms"}:
            raise ValueError("AI_GENERATION_PROVIDER must be mock or gms")
        app_env = os.getenv("APP_ENV", "development").lower()
        internal_api_key = os.getenv(
            "AI_INTERNAL_API_KEY", "local-development-key"
        )
        gms_key = os.getenv("GMS_KEY", "")
        if app_env not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production")
        if not internal_api_key:
            raise ValueError("AI_INTERNAL_API_KEY must not be empty")
        if app_env == "production" and internal_api_key == "local-development-key":
            raise ValueError("set a unique AI_INTERNAL_API_KEY in production")
        if app_env == "production" and provider == "mock":
            raise ValueError("AI_GENERATION_PROVIDER=mock is not allowed in production")
        if provider == "gms" and not gms_key:
            raise ValueError("GMS_KEY is required when AI_GENERATION_PROVIDER=gms")
        text_timeout = float(os.getenv("AI_GMS_TEXT_TIMEOUT_SECONDS", "28"))
        output_tokens = int(os.getenv("AI_GMS_MAX_OUTPUT_TOKENS", "3200"))
        idempotency_ttl = float(os.getenv("AI_IDEMPOTENCY_TTL_SECONDS", "600"))
        if not math.isfinite(text_timeout) or text_timeout <= 0:
            raise ValueError("AI_GMS_TEXT_TIMEOUT_SECONDS must be positive and finite")
        if output_tokens < 256:
            raise ValueError("AI_GMS_MAX_OUTPUT_TOKENS must be at least 256")
        if not math.isfinite(idempotency_ttl) or idempotency_ttl <= 0:
            raise ValueError("AI_IDEMPOTENCY_TTL_SECONDS must be positive and finite")
        return cls(
            internal_api_key=internal_api_key,
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION", ""),
            azure_speech_language=os.getenv("AZURE_SPEECH_LANGUAGE", "ko-KR"),
            max_audio_bytes=max_audio_bytes,
            app_env=app_env,
            generation_provider=provider,
            gms_key=gms_key,
            gms_base_url=os.getenv("GMS_BASE_URL", "https://gms.ssafy.io/gmsapi"),
            gms_openai_base_url=os.getenv("GMS_OPENAI_BASE_URL", ""),
            gms_text_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            gms_text_timeout_seconds=text_timeout,
            gms_max_output_tokens=output_tokens,
            idempotency_ttl_seconds=idempotency_ttl,
        )

    @property
    def gms_text_base_url(self) -> str:
        if self.gms_openai_base_url:
            return self.gms_openai_base_url.rstrip("/")
        return f"{self.gms_base_url.rstrip('/')}/api.openai.com/v1"
