from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iread_ai.contracts.comparison import (
    DisplayedChapterComparisonRequest,
)
from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)


def build_displayed_chapter_comparison_payload(
    chapter_request_payload: Mapping[str, Any],
    personalized_response_payload: Mapping[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    payload = DisplayedChapterComparisonRequest(
        requestId=request_id,
        chapterRequest=StoryChapterGenerateRequest.model_validate(
            dict(chapter_request_payload)
        ),
        personalizedResponse=StoryChapterGenerateResponse.model_validate(
            dict(personalized_response_payload)
        ),
    )
    return payload.model_dump(mode="json", by_alias=True)


__all__ = ["build_displayed_chapter_comparison_payload"]
