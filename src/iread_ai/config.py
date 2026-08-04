from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    internal_api_key: SecretStr = Field(
        default=SecretStr("local-development-key"),
        validation_alias=AliasChoices(
            "AI_INTERNAL_API_KEY",
            "INTERNAL_API_KEY",
        ),
    )

    azure_speech_key: str = ""
    azure_speech_region: str = ""
    azure_speech_language: str = "ko-KR"
    azure_speech_voice: str = "ko-KR-SunHiNeural"
    pronunciation_provider: Literal["azure", "gms", "deterministic"] = Field(
        default="deterministic",
        validation_alias="AI_PRONUNCIATION_PROVIDER",
    )
    speech_provider: Literal["azure", "deterministic"] = Field(
        default="deterministic",
        validation_alias="AI_SPEECH_PROVIDER",
    )
    max_audio_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
        validation_alias="AI_MAX_AUDIO_BYTES",
    )

    story_provider: Literal["mock", "openai", "gms"] = "mock"
    generation_provider: Literal["mock", "openai", "gms"] = Field(
        default="mock",
        validation_alias="AI_GENERATION_PROVIDER",
    )
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_max_output_tokens: int = Field(default=2400, ge=256, le=16_384)
    openai_chapter_max_output_tokens: int = Field(
        default=8000,
        ge=1024,
        le=16_384,
    )
    openai_visual_scene_max_output_tokens: int = Field(
        default=3200,
        ge=512,
        le=16_384,
    )
    visual_scene_timeout_seconds: float = Field(
        default=8.0,
        gt=0,
        le=20.0,
    )
    chapter_candidate_count: int = Field(default=1, ge=1, le=8)

    gms_key: SecretStr | None = None
    gms_base_url: str = "https://gms.ssafy.io/gmsapi"
    gms_openai_base_url: str | None = None
    story_image_provider: Literal["disabled", "gemini"] = "disabled"
    gms_gemini_image_model: str = "gemini-2.5-flash-image"
    gms_image_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
    gms_image_max_bytes: int = Field(
        default=12 * 1024 * 1024,
        ge=1024,
        le=12 * 1024 * 1024,
    )
    gms_image_max_response_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        le=32 * 1024 * 1024,
    )
    gms_image_max_request_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        le=32 * 1024 * 1024,
    )
    character_reference_dir: Path = Path("assets/character-references")
    lexicon_database_path: Path = Field(
        default=Path("local-output/lexicon/story-lexicon.sqlite3"),
        validation_alias="AI_LEXICON_DB_PATH",
    )

    model_timeout_seconds: float = Field(default=75.0, gt=0, le=120.0)
    idempotency_ttl_seconds: float = Field(
        default=600.0,
        gt=0,
        validation_alias=AliasChoices(
            "AI_IDEMPOTENCY_TTL_SECONDS",
            "IDEMPOTENCY_TTL_SECONDS",
        ),
    )
    gms_text_timeout_seconds: float = Field(
        default=28.0,
        gt=0,
        validation_alias="AI_GMS_TEXT_TIMEOUT_SECONDS",
    )
    gms_speech_model: str = Field(
        default="whisper-1",
        min_length=1,
        validation_alias="AI_GMS_SPEECH_MODEL",
    )
    gms_speech_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        le=120,
        validation_alias="AI_GMS_SPEECH_TIMEOUT_SECONDS",
    )
    gms_max_output_tokens: int = Field(
        default=3200,
        ge=256,
        le=16_384,
        validation_alias="AI_GMS_MAX_OUTPUT_TOKENS",
    )

    @classmethod
    def from_env(cls) -> Settings:
        return cls()

    @model_validator(mode="after")
    def validate_runtime(self) -> Settings:
        internal_key = self.internal_api_key.get_secret_value()
        if not internal_key:
            raise ValueError("AI_INTERNAL_API_KEY must not be empty")
        if self.app_env == "production" and internal_key == "local-development-key":
            raise ValueError("Set a unique AI_INTERNAL_API_KEY in production")
        if self.app_env == "production" and self.story_provider == "mock":
            raise ValueError("STORY_PROVIDER=mock is not allowed in production")
        if self.app_env == "production" and self.generation_provider == "mock":
            raise ValueError(
                "AI_GENERATION_PROVIDER=mock is not allowed in production"
            )
        if self.story_provider == "openai" and (
            self.openai_api_key is None
            or not self.openai_api_key.get_secret_value()
        ):
            raise ValueError(
                "OPENAI_API_KEY is required when STORY_PROVIDER=openai"
            )
        if self.generation_provider == "openai" and (
            self.openai_api_key is None
            or not self.openai_api_key.get_secret_value()
        ):
            raise ValueError(
                "OPENAI_API_KEY is required when AI_GENERATION_PROVIDER=openai"
            )
        if self.story_provider == "gms" and (
            self.gms_key is None or not self.gms_key.get_secret_value()
        ):
            raise ValueError("GMS_KEY is required when STORY_PROVIDER=gms")
        if self.generation_provider == "gms" and (
            self.gms_key is None or not self.gms_key.get_secret_value()
        ):
            raise ValueError(
                "GMS_KEY is required when AI_GENERATION_PROVIDER=gms"
            )
        if self.pronunciation_provider == "gms" and (
            self.gms_key is None or not self.gms_key.get_secret_value()
        ):
            raise ValueError(
                "GMS_KEY is required when AI_PRONUNCIATION_PROVIDER=gms"
            )
        if self.story_image_provider == "gemini" and (
            self.gms_key is None or not self.gms_key.get_secret_value()
        ):
            raise ValueError(
                "GMS_KEY is required when STORY_IMAGE_PROVIDER=gemini"
            )
        return self

    @property
    def story_api_key(self) -> str:
        if self.story_provider == "gms":
            assert self.gms_key is not None
            return self.gms_key.get_secret_value()
        if self.story_provider == "openai":
            assert self.openai_api_key is not None
            return self.openai_api_key.get_secret_value()
        raise RuntimeError("mock story generation does not use an API key")

    @property
    def story_api_base_url(self) -> str:
        if self.story_provider == "gms":
            if self.gms_openai_base_url:
                return self.gms_openai_base_url.rstrip("/")
            return (
                f"{self.gms_base_url.rstrip('/')}"
                "/api.openai.com/v1"
            )
        if self.story_provider == "openai":
            return self.openai_base_url.rstrip("/")
        raise RuntimeError("mock story generation does not use an API URL")

    @property
    def gms_text_model(self) -> str:
        return self.openai_model

    @property
    def gms_text_base_url(self) -> str:
        if self.gms_openai_base_url:
            return self.gms_openai_base_url.rstrip("/")
        return f"{self.gms_base_url.rstrip('/')}/api.openai.com/v1"

    @property
    def text_api_key(self) -> str:
        if self.generation_provider == "openai":
            assert self.openai_api_key is not None
            return self.openai_api_key.get_secret_value()
        if self.generation_provider == "gms":
            assert self.gms_key is not None
            return self.gms_key.get_secret_value()
        raise RuntimeError("mock text generation does not use an API key")

    @property
    def text_base_url(self) -> str:
        if self.generation_provider == "openai":
            return self.openai_base_url.rstrip("/")
        if self.generation_provider == "gms":
            return self.gms_text_base_url
        raise RuntimeError("mock text generation does not use an API URL")

    @property
    def text_timeout_seconds(self) -> float:
        if self.generation_provider == "openai":
            return self.model_timeout_seconds
        return self.gms_text_timeout_seconds

    @property
    def text_max_output_tokens(self) -> int:
        if self.generation_provider == "openai":
            return self.openai_max_output_tokens
        return self.gms_max_output_tokens


@lru_cache
def get_settings() -> Settings:
    return Settings()
