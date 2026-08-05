from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoryImageReference:
    character_id: str
    name: str
    description: str
    content: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class GeneratedStoryImage:
    content: bytes
    mime_type: str
    provider_request_id: str | None = None


class StoryImageProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable


class StoryImageGenerator(Protocol):
    @property
    def model(self) -> str: ...

    async def generate(
        self,
        *,
        prompt: str,
        references: Sequence[StoryImageReference],
        aspect_ratio: str,
    ) -> GeneratedStoryImage: ...


__all__ = [
    "GeneratedStoryImage",
    "StoryImageGenerator",
    "StoryImageProviderError",
    "StoryImageReference",
]
