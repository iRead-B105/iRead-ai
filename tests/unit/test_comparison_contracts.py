from __future__ import annotations

import pytest
from pydantic import ValidationError

from iread_ai.contracts.comparison import DisplayedChapterComparisonRequest
from tests.unit.test_story_chapter_contracts import (
    request_payload as chapter_request_payload,
)
from tests.unit.test_story_chapter_contracts import (
    response_payload as chapter_response_payload,
)


def test_displayed_chapter_comparison_requires_an_exact_request_pair() -> None:
    comparison = DisplayedChapterComparisonRequest.model_validate(
        {
            "requestId": "chapter-comparison-1",
            "chapterRequest": chapter_request_payload(),
            "personalizedResponse": chapter_response_payload(),
        }
    )

    assert comparison.chapter_request.chapter_number == 2
    assert comparison.personalized_response.generation.candidate_count == 3

    mismatched = chapter_response_payload()
    mismatched["storyRevision"] += 1
    with pytest.raises(ValidationError, match="storyRevision"):
        DisplayedChapterComparisonRequest.model_validate(
            {
                "requestId": "chapter-comparison-mismatch",
                "chapterRequest": chapter_request_payload(),
                "personalizedResponse": mismatched,
            }
        )


def test_displayed_chapter_comparison_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DisplayedChapterComparisonRequest.model_validate(
            {
                "requestId": "chapter-comparison-1",
                "chapterRequest": chapter_request_payload(),
                "personalizedResponse": chapter_response_payload(),
                "unknownField": True,
            }
        )
