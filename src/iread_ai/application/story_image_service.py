from __future__ import annotations

import base64
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from iread_ai.contracts.story_image import (
    StoryImageGenerateRequest,
    StoryImageGenerateResponse,
)
from iread_ai.personalization.story_image_prompt import (
    build_story_image_prompt,
    story_image_prompt_version,
)
from iread_ai.ports.story_image_generator import (
    StoryImageGenerator,
    StoryImageProviderError,
    StoryImageReference,
)


@dataclass(frozen=True, slots=True)
class CharacterReferenceDefinition:
    name: str
    description: str
    filename: str


DEFAULT_CHARACTER_REFERENCES: Mapping[str, CharacterReferenceDefinition] = {
    "hare": CharacterReferenceDefinition(
        name="토끼",
        description=(
            "a small white storybook rabbit with black oval eyes, a pink-red "
            "nose, pink inner ears, one upright ear and one ear bent sideways"
        ),
        filename="rabbit.png",
    ),
    "tortoise": CharacterReferenceDefinition(
        name="거북이",
        description=(
            "a mint-green baby storybook tortoise with a round head, large "
            "black eyes, pale yellow belly plates, and a green patterned shell"
        ),
        filename="turtle.png",
    ),
}


class StoryImageReferenceRepository(Protocol):
    def resolve(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]: ...

    def resolve_available(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]: ...


class CharacterReferenceNotFoundError(LookupError):
    pass


class CharacterReferenceConfigurationError(RuntimeError):
    pass


class KnownCharacterReferenceRepository:
    def __init__(
        self,
        *,
        root: Path | None = None,
        definitions: Mapping[
            str,
            CharacterReferenceDefinition,
        ] = DEFAULT_CHARACTER_REFERENCES,
        max_image_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self._root = (
            root
            or Path(__file__).resolve().parents[3]
            / "assets"
            / "character-references"
        ).resolve()
        self._definitions = dict(definitions)
        self._max_image_bytes = max_image_bytes
        if max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")

    def resolve(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        references: list[StoryImageReference] = []
        for character_id in character_ids:
            definition = self._definitions.get(character_id)
            if definition is None:
                raise CharacterReferenceNotFoundError(character_id)
            source = (self._root / definition.filename).resolve()
            if source.parent != self._root:
                raise CharacterReferenceConfigurationError(
                    "character reference path escaped the configured asset root"
                )
            try:
                content = source.read_bytes()
            except OSError as exc:
                raise CharacterReferenceConfigurationError(
                    f"character reference asset is unavailable: {character_id}"
                ) from exc
            if not content or len(content) > self._max_image_bytes:
                raise CharacterReferenceConfigurationError(
                    f"character reference asset is invalid: {character_id}"
                )
            mime_type = _detect_image_mime(content)
            if mime_type is None:
                raise CharacterReferenceConfigurationError(
                    f"character reference asset has an unsupported type: {character_id}"
                )
            references.append(
                StoryImageReference(
                    character_id=character_id,
                    name=definition.name,
                    description=definition.description,
                    content=content,
                    mime_type=mime_type,
                )
            )
        return tuple(references)

    def resolve_available(
        self,
        character_ids: Sequence[str],
    ) -> tuple[StoryImageReference, ...]:
        references: list[StoryImageReference] = []
        for character_id in character_ids:
            if character_id not in self._definitions:
                continue
            try:
                references.extend(self.resolve([character_id]))
            except CharacterReferenceConfigurationError:
                continue
        return tuple(references)


class StoryImageUseCaseError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable


class StoryImageApplicationService:
    def __init__(
        self,
        *,
        generator: StoryImageGenerator | None,
        references: StoryImageReferenceRepository | None = None,
    ) -> None:
        self._generator = generator
        self._references = references or KnownCharacterReferenceRepository()

    async def generate(
        self,
        request: StoryImageGenerateRequest,
    ) -> StoryImageGenerateResponse:
        if self._generator is None:
            raise StoryImageUseCaseError(
                status_code=503,
                code="IMAGE_PROVIDER_NOT_CONFIGURED",
                message="이미지 생성 공급자가 설정되지 않았습니다.",
                retryable=False,
            )
        reference_ids = [
            reference.character_id
            for reference in request.character_references
        ]
        try:
            if reference_ids:
                resolved_references = self._references.resolve(reference_ids)
            else:
                present_character_ids = [
                    character.character_id
                    for character in request.visual_scene.characters
                    if character.present
                ]
                resolved_references = self._references.resolve_available(
                    present_character_ids
                )
        except CharacterReferenceNotFoundError as exc:
            raise StoryImageUseCaseError(
                status_code=422,
                code="UNKNOWN_CHARACTER_REFERENCE",
                message=(
                    "서버에 등록되지 않은 캐릭터 레퍼런스입니다: "
                    f"{exc.args[0]}"
                ),
                retryable=False,
            ) from exc
        except CharacterReferenceConfigurationError as exc:
            raise StoryImageUseCaseError(
                status_code=503,
                code="CHARACTER_REFERENCE_UNAVAILABLE",
                message="캐릭터 레퍼런스 에셋을 불러오지 못했습니다.",
                retryable=True,
            ) from exc

        prompt = build_story_image_prompt(
            request=request,
            references=resolved_references,
        )
        started = time.perf_counter()
        try:
            generated = await self._generator.generate(
                prompt=prompt,
                references=resolved_references,
                aspect_ratio="21:9",
            )
        except StoryImageProviderError as exc:
            raise _map_provider_error(exc) from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        response = StoryImageGenerateResponse(
            requestId=request.request_id,
            schemaVersion=1,
            imageId=f"image-{uuid4().hex}",
            mimeType=generated.mime_type,
            imageBase64=base64.b64encode(generated.content).decode("ascii"),
            model=self._generator.model,
            promptVersion=story_image_prompt_version(),
            timingMs=elapsed_ms,
        )
        return response.validate_against_request(request)


def _map_provider_error(
    error: StoryImageProviderError,
) -> StoryImageUseCaseError:
    if error.code == "TIMEOUT":
        return StoryImageUseCaseError(
            status_code=504,
            code="IMAGE_GENERATION_TIMEOUT",
            message="이미지 생성 시간이 초과되었습니다.",
            retryable=True,
        )
    if error.code == "SAFETY_BLOCKED":
        return StoryImageUseCaseError(
            status_code=422,
            code="IMAGE_GENERATION_BLOCKED",
            message="안전 정책으로 인해 이 장면의 이미지를 생성하지 못했습니다.",
            retryable=False,
        )
    return StoryImageUseCaseError(
        status_code=502,
        code="IMAGE_PROVIDER_ERROR",
        message="이미지 생성 모델의 응답을 처리하지 못했습니다.",
        retryable=error.retryable,
    )


def _detect_image_mime(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if (
        len(content) >= 12
        and content.startswith(b"RIFF")
        and content[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


__all__ = [
    "CharacterReferenceConfigurationError",
    "CharacterReferenceDefinition",
    "CharacterReferenceNotFoundError",
    "KnownCharacterReferenceRepository",
    "StoryImageApplicationService",
    "StoryImageReferenceRepository",
    "StoryImageUseCaseError",
]
