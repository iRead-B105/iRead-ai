from __future__ import annotations

import json
from pathlib import Path

from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.contracts.story_image import StoryImageGenerateRequest

ROOT = Path(__file__).resolve().parents[2]


def _read_json(relative_path: str) -> object:
    return json.loads(
        (ROOT / relative_path).read_text(encoding="utf-8")
    )


def test_documented_opening_and_continuation_requests_are_valid() -> None:
    opening = StoryChapterGenerateRequest.model_validate(
        _read_json(
            "docs/examples/story-chapter-opening-request.json"
        )
    )
    continuation = StoryChapterGenerateRequest.model_validate(
        _read_json(
            "docs/examples/story-chapter-continuation-request.json"
        )
    )

    assert opening.chapter_number == 1
    assert opening.story_revision == 0
    assert opening.branch_input is None
    assert continuation.chapter_number == 2
    assert continuation.branch_input is not None


def test_documented_image_request_is_valid_without_reference_assets() -> None:
    request = StoryImageGenerateRequest.model_validate(
        _read_json("docs/examples/story-image-request.json")
    )

    assert request.character_references == []
