from __future__ import annotations

import pytest
from pydantic import ValidationError

from iread_ai.config import Settings


def test_gms_story_provider_uses_shared_key_and_derived_proxy_url() -> None:
    settings = Settings(
        _env_file=None,
        generation_provider="mock",
        story_provider="gms",
        gms_key="gms-test-key",
        gms_base_url="https://gateway.example/gmsapi/",
        openai_api_key=None,
    )

    assert settings.story_api_key == "gms-test-key"
    assert settings.story_api_base_url == "https://gateway.example/gmsapi/api.openai.com/v1"
    assert "gms-test-key" not in repr(settings)


def test_gms_story_provider_accepts_explicit_openai_compatible_url() -> None:
    settings = Settings(
        _env_file=None,
        generation_provider="mock",
        story_provider="gms",
        gms_key="gms-test-key",
        gms_openai_base_url="https://gateway.example/openai/v1/",
        openai_api_key=None,
    )

    assert settings.story_api_base_url == "https://gateway.example/openai/v1"


def test_gms_story_provider_requires_gms_key() -> None:
    with pytest.raises(
        ValidationError,
        match="GMS_KEY is required when STORY_PROVIDER=gms",
    ):
        Settings(
            _env_file=None,
            generation_provider="mock",
            story_provider="gms",
            gms_key=None,
            openai_api_key=None,
        )


def test_direct_openai_story_provider_remains_supported() -> None:
    settings = Settings(
        _env_file=None,
        story_provider="openai",
        openai_api_key="openai-test-key",
        openai_base_url="https://openai.example/v1/",
        gms_key=None,
    )

    assert settings.story_api_key == "openai-test-key"
    assert settings.story_api_base_url == "https://openai.example/v1"


def test_story_image_generation_is_disabled_by_default_with_gms_key() -> None:
    settings = Settings(
        _env_file=None,
        story_provider="gms",
        gms_key="gms-test-key",
    )

    assert settings.story_image_provider == "disabled"


def test_direct_openai_training_provider_uses_openai_credentials() -> None:
    settings = Settings(
        _env_file=None,
        generation_provider="openai",
        openai_api_key="openai-test-key",
        openai_base_url="https://openai.example/v1/",
        gms_key=None,
    )

    assert settings.text_api_key == "openai-test-key"
    assert settings.text_base_url == "https://openai.example/v1"
    assert settings.text_timeout_seconds == settings.model_timeout_seconds
    assert settings.text_max_output_tokens == settings.openai_max_output_tokens


def test_direct_openai_training_provider_requires_openai_key() -> None:
    with pytest.raises(
        ValidationError,
        match="OPENAI_API_KEY is required when AI_GENERATION_PROVIDER=openai",
    ):
        Settings(
            _env_file=None,
            generation_provider="openai",
            story_provider="mock",
            openai_api_key=None,
            gms_key=None,
        )


def test_gemini_story_image_provider_requires_gemini_key() -> None:
    with pytest.raises(
        ValidationError,
        match="GEMINI_API_KEY is required when STORY_IMAGE_PROVIDER=gemini",
    ):
        Settings(
            _env_file=None,
            story_image_provider="gemini",
            gemini_api_key=None,
        )
