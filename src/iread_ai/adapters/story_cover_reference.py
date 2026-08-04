from __future__ import annotations

from pathlib import Path

from iread_ai.ports.story_image_generator import StoryImageProviderError, StoryImageReference

_COVERS = {
    1: "rabbit-and-turtle.png",
    2: "ant-and-grasshopper.png",
    3: "old-man-and-sea.png",
    4: "cinderella.png",
    5: "byeoljubujeon.png",
    6: "three-little-pigs.png",
}


class StoryCoverReferenceRepository:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def resolve(self, template_id: int | None) -> tuple[StoryImageReference, ...]:
        filename = _COVERS.get(template_id or 0)
        if filename is None:
            raise StoryImageProviderError(
                "COVER_NOT_FOUND", "Story cover reference is not configured.", retryable=False
            )
        source = (self._root / filename).resolve()
        if source.parent != self._root:
            raise StoryImageProviderError(
                "INVALID_COVER", "Story cover reference path is invalid.", retryable=False
            )
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise StoryImageProviderError(
                "COVER_UNAVAILABLE", "Story cover reference is unavailable.", retryable=False
            ) from exc
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise StoryImageProviderError(
                "INVALID_COVER", "Story cover reference must be a PNG image.", retryable=False
            )
        return (
            StoryImageReference(
                character_id=f"story-template-{template_id}",
                name="approved book cover",
                description=(
                    "Use this cover as the visual style and character consistency reference."
                ),
                content=content,
                mime_type="image/png",
            ),
        )


__all__ = ["StoryCoverReferenceRepository"]
