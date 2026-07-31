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
    max_audio_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
        validation_alias="AI_MAX_AUDIO_BYTES",
    )

    story_provider: Literal["mock", "openai", "gms"] = "mock"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_max_output_tokens: int = Field(default=2400, ge=256, le=16_384)
    openai_chapter_max_output_tokens: int = Field(
        default=5000,
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
    chapter_candidate_count: int = Field(default=3, ge=1, le=8)

    gms_key: SecretStr | None = None
    gms_base_url: str = "https://gms.ssafy.io/gmsapi"
    gms_openai_base_url: str | None = None
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

    model_timeout_seconds: float = Field(default=28.0, gt=0, le=29.0)
    idempotency_ttl_seconds: float = Field(default=600.0, gt=0)

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
        if self.story_provider == "openai" and (
            self.openai_api_key is None
            or not self.openai_api_key.get_secret_value()
        ):
            raise ValueError(
                "OPENAI_API_KEY is required when STORY_PROVIDER=openai"
            )
        if self.story_provider == "gms" and (
            self.gms_key is None or not self.gms_key.get_secret_value()
        ):
            raise ValueError("GMS_KEY is required when STORY_PROVIDER=gms")
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
