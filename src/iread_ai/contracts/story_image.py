from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictFloat, StrictInt, StrictStr, StringConstraints, model_validator

from iread_ai.contracts.story_chapter import StoryVisualScenePayload
from iread_ai.contracts.story_page import (
    NonEmptyText,
    ShortIdentifier,
    StoryCharacterPayload,
    StoryPageContractModel,
)

ImageBase64 = Annotated[
    StrictStr,
    StringConstraints(min_length=1, max_length=16 * 1024 * 1024),
]
StoryImageMimeType = Literal["image/png", "image/jpeg", "image/webp"]


class StoryImageContextPayload(StoryPageContractModel):
    title: NonEmptyText
    characters: list[StoryCharacterPayload] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_unique_characters(self) -> Self:
        character_ids = [character.character_id for character in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("storyContext characterId values must be unique")
        return self


class StoryImageCharacterReferencePayload(StoryPageContractModel):
    character_id: ShortIdentifier


class StoryImageGenerateRequest(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[1]
    story_id: StrictInt = Field(ge=1)
    story_revision: StrictInt = Field(ge=0)
    chapter_number: StrictInt = Field(ge=1)
    page_number: StrictInt = Field(ge=1, le=4)
    sentences: list[NonEmptyText] = Field(min_length=3, max_length=4)
    visual_scene: StoryVisualScenePayload
    story_context: StoryImageContextPayload
    character_references: list[StoryImageCharacterReferencePayload] = Field(
        default_factory=list,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_character_links(self) -> Self:
        context_ids = {character.character_id for character in self.story_context.characters}
        scene_ids = {character.character_id for character in self.visual_scene.characters}
        unknown_scene_ids = scene_ids - context_ids
        if unknown_scene_ids:
            raise ValueError(
                "visualScene contains unknown characterId values: "
                + ", ".join(sorted(unknown_scene_ids))
            )

        reference_ids = [reference.character_id for reference in self.character_references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("characterReferences characterId values must be unique")
        unknown_reference_ids = set(reference_ids) - context_ids
        if unknown_reference_ids:
            raise ValueError(
                "characterReferences contains unknown characterId values: "
                + ", ".join(sorted(unknown_reference_ids))
            )
        return self


class StoryImageGenerateResponse(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[1]
    image_id: ShortIdentifier
    mime_type: StoryImageMimeType
    image_base64: ImageBase64
    model: NonEmptyText
    prompt_version: ShortIdentifier
    timing_ms: StrictFloat = Field(ge=0)

    def validate_against_request(
        self,
        request: StoryImageGenerateRequest,
    ) -> Self:
        if self.request_id != request.request_id:
            raise ValueError("response requestId must match request requestId")
        return self


__all__ = [
    "StoryImageCharacterReferencePayload",
    "StoryImageContextPayload",
    "StoryImageGenerateRequest",
    "StoryImageGenerateResponse",
    "StoryImageMimeType",
]
