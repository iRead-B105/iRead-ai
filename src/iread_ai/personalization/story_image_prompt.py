from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from iread_ai.contracts.story_image import StoryImageGenerateRequest
from iread_ai.ports.story_image_generator import StoryImageReference

DEFAULT_STORY_IMAGE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "story_image.md"
_PROMPT_SCHEMA_VERSION = "story-image-prompt-v1"


def load_story_image_prompt(path: Path | None = None) -> str:
    prompt_path = path or DEFAULT_STORY_IMAGE_PROMPT_PATH
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("story image prompt must not be empty")
    if len(prompt.encode("utf-8")) > 48 * 1024:
        raise ValueError("story image prompt must not exceed 48 KiB")
    return prompt


def story_image_prompt_version(path: Path | None = None) -> str:
    prompt = load_story_image_prompt(path)
    digest = hashlib.sha256(f"{_PROMPT_SCHEMA_VERSION}\n{prompt}".encode()).hexdigest()[:12]
    return f"story-image-{digest}"


def build_story_image_prompt(
    *,
    request: StoryImageGenerateRequest,
    references: Sequence[StoryImageReference],
    prompt_path: Path | None = None,
) -> str:
    policy = load_story_image_prompt(prompt_path)
    reference_lines = [
        (
            f"- Input image {index}: characterId={reference.character_id}; "
            f"name={reference.name}; identity={reference.description}"
        )
        for index, reference in enumerate(references, start=1)
    ]
    if not reference_lines:
        reference_lines = [
            "- No identity image is attached. Use the character catalog's "
            "immutableTraits as the identity source."
        ]
    story_context = request.story_context.model_dump(
        mode="json",
        by_alias=True,
    )
    visual_scene = request.visual_scene.model_dump(
        mode="json",
        by_alias=True,
    )
    sentences = "\n".join(
        f"{index}. {sentence}" for index, sentence in enumerate(request.sentences, start=1)
    )
    return (
        f"{policy}\n\n"
        "[ATTACHED IDENTITY IMAGES IN EXACT ORDER]\n"
        f"{chr(10).join(reference_lines)}\n\n"
        "[STORY CHARACTER CATALOG]\n"
        f"{_compact_json(story_context)}\n\n"
        "[CURRENT PAGE VISUALSCENE — AUTHORITATIVE]\n"
        f"{_compact_json(visual_scene)}\n\n"
        f"[CURRENT PAGE {request.page_number} SENTENCES]\n"
        f"{sentences}\n\n"
        "Rebuild the composition for this page and faithfully apply visualScene. "
        "Return one continuous full-bleed 21:9 image."
    )


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "DEFAULT_STORY_IMAGE_PROMPT_PATH",
    "build_story_image_prompt",
    "load_story_image_prompt",
    "story_image_prompt_version",
]
