from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path

import pytest

from iread_ai.application.story_image_service import (
    CharacterReferenceNotFoundError,
    KnownCharacterReferenceRepository,
    StoryImageApplicationService,
    StoryImageUseCaseError,
)
from iread_ai.contracts.story_image import StoryImageGenerateRequest
from iread_ai.ports.story_image_generator import (
    GeneratedStoryImage,
    StoryImageProviderError,
    StoryImageReference,
)
from tests.unit.test_story_image_contracts import image_request_payload

_VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class RecordingImageGenerator:
    model = "gemini-2.5-flash-image"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        *,
        prompt: str,
        references: Sequence[StoryImageReference],
        aspect_ratio: str,
    ) -> GeneratedStoryImage:
        self.calls.append(
            {
                "prompt": prompt,
                "references": tuple(references),
                "aspect_ratio": aspect_ratio,
            }
        )
        return GeneratedStoryImage(
            content=_VALID_PNG,
            mime_type="image/png",
        )


class EmptyReferenceRepository:
    def resolve(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        assert list(character_ids) == []
        return ()

    def resolve_available(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        assert list(character_ids) == ["hare"]
        return ()


class MissingReferenceRepository:
    def resolve(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        raise CharacterReferenceNotFoundError(character_ids[0])

    def resolve_available(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        return ()


class AutomaticReferenceRepository:
    def resolve(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        raise AssertionError("empty client override must use resolve_available")

    def resolve_available(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        assert list(character_ids) == ["hare"]
        return (
            StoryImageReference(
                character_id="hare",
                name="토끼",
                description="white rabbit identity",
                content=_VALID_PNG,
                mime_type="image/png",
            ),
        )


def test_known_reference_repository_uses_configured_asset_root(
    tmp_path: Path,
) -> None:
    (tmp_path / "rabbit.png").write_bytes(_VALID_PNG)
    repository = KnownCharacterReferenceRepository(root=tmp_path)

    references = repository.resolve(["hare"])

    assert len(references) == 1
    assert references[0].character_id == "hare"
    assert references[0].content == _VALID_PNG


def test_missing_optional_reference_assets_are_skipped(
    tmp_path: Path,
) -> None:
    repository = KnownCharacterReferenceRepository(root=tmp_path)

    assert repository.resolve_available(["hare", "tortoise"]) == ()


@pytest.mark.asyncio
async def test_service_uses_llm_visual_scene_and_allows_no_identity_asset() -> None:
    payload = image_request_payload()
    payload["characterReferences"] = []
    request = StoryImageGenerateRequest.model_validate(payload)
    generator = RecordingImageGenerator()
    service = StoryImageApplicationService(
        generator=generator,
        references=EmptyReferenceRepository(),
    )

    response = await service.generate(request)

    assert response.mime_type == "image/png"
    assert response.model == "gemini-2.5-flash-image"
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["aspect_ratio"] == "21:9"
    assert call["references"] == ()
    prompt = str(call["prompt"])
    assert '"emotion":{"intensity":"MEDIUM","type":"EXCITED"}' in prompt
    assert "Use the full canvas naturally" in prompt
    assert "rightmost 70%" not in prompt
    assert "leftmost 30%" not in prompt
    assert "Do not reserve, empty, blur, fade, or simplify any fixed side" in prompt
    assert "Never create a split screen, panel, card, picture frame" in prompt
    assert "No identity image is attached" in prompt


@pytest.mark.asyncio
async def test_empty_client_reference_list_auto_selects_known_present_assets() -> None:
    payload = image_request_payload()
    payload["characterReferences"] = []
    request = StoryImageGenerateRequest.model_validate(payload)
    generator = RecordingImageGenerator()
    service = StoryImageApplicationService(
        generator=generator,
        references=AutomaticReferenceRepository(),
    )

    await service.generate(request)

    sent_references = generator.calls[0]["references"]
    assert isinstance(sent_references, tuple)
    assert [reference.character_id for reference in sent_references] == ["hare"]
    assert "Input image 1: characterId=hare" in str(generator.calls[0]["prompt"])


@pytest.mark.asyncio
async def test_unknown_requested_reference_is_a_clear_422() -> None:
    request = StoryImageGenerateRequest.model_validate(image_request_payload())
    service = StoryImageApplicationService(
        generator=RecordingImageGenerator(),
        references=MissingReferenceRepository(),
    )

    with pytest.raises(StoryImageUseCaseError) as raised:
        await service.generate(request)

    assert raised.value.status_code == 422
    assert raised.value.code == "UNKNOWN_CHARACTER_REFERENCE"
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_unconfigured_image_provider_is_a_clear_503() -> None:
    request = StoryImageGenerateRequest.model_validate(image_request_payload())
    service = StoryImageApplicationService(
        generator=None,
        references=MissingReferenceRepository(),
    )

    with pytest.raises(StoryImageUseCaseError) as raised:
        await service.generate(request)

    assert raised.value.status_code == 503
    assert raised.value.code == "IMAGE_PROVIDER_NOT_CONFIGURED"


class TimeoutImageGenerator(RecordingImageGenerator):
    async def generate(
        self,
        *,
        prompt: str,
        references: Sequence[StoryImageReference],
        aspect_ratio: str,
    ) -> GeneratedStoryImage:
        raise StoryImageProviderError(
            "TIMEOUT",
            "safe timeout",
            retryable=True,
        )


@pytest.mark.asyncio
async def test_provider_timeout_maps_to_retryable_504() -> None:
    payload = image_request_payload()
    payload["characterReferences"] = []
    request = StoryImageGenerateRequest.model_validate(payload)
    service = StoryImageApplicationService(
        generator=TimeoutImageGenerator(),
        references=EmptyReferenceRepository(),
    )

    with pytest.raises(StoryImageUseCaseError) as raised:
        await service.generate(request)

    assert raised.value.status_code == 504
    assert raised.value.retryable is True
