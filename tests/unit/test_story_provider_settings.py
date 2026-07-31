from __future__ import annotations

import pytest
from pydantic import ValidationError

from iread_ai.config import Settings


def test_gms_story_provider_uses_shared_key_and_derived_proxy_url() -> None:
    settings = Settings(
        _env_file=None,
        story_provider="gms",
        gms_key="gms-test-key",
        gms_base_url="https://gateway.example/gmsapi/",
        openai_api_key=None,
    )

    assert settings.story_api_key == "gms-test-key"
    assert (
        settings.story_api_base_url
        == "https://gateway.example/gmsapi/api.openai.com/v1"
    )
    assert "gms-test-key" not in repr(settings)


def test_gms_story_provider_accepts_explicit_openai_compatible_url() -> None:
    settings = Settings(
        _env_file=None,
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
