from __future__ import annotations

from iread_ai.application.chapter_comparison_builder import (
    build_displayed_chapter_comparison_payload,
)
from tests.unit.test_story_chapter_contracts import (
    request_payload,
    response_payload,
)


def test_builder_pairs_the_exact_v3_request_and_displayed_response() -> None:
    payload = build_displayed_chapter_comparison_payload(
        request_payload(),
        response_payload(),
        request_id="displayed-chapter-pair",
    )

    assert payload["requestId"] == "displayed-chapter-pair"
    assert payload["chapterRequest"]["requestId"] == (payload["personalizedResponse"]["requestId"])
    assert (
        payload["chapterRequest"]["storyRevision"]
        == (payload["personalizedResponse"]["storyRevision"])
    )
    assert (
        payload["chapterRequest"]["chapterNumber"]
        == (payload["personalizedResponse"]["chapterNumber"])
    )
    assert payload["personalizedResponse"]["generation"]["candidateCount"] == 3
