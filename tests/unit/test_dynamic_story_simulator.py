from __future__ import annotations

import copy
from typing import Any

import pytest

from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.devtools.dynamic_story_simulator import (
    ChapterGenerationCursor,
    DynamicStoryStateError,
    advance_display_cursor,
    apply_chapter_response,
    build_chapter_request,
    displayed_page,
    generation_cursor_after_response,
    initial_display_cursor,
    initial_dynamic_runtime,
    initial_generation_cursor,
)
from iread_ai.devtools.service_story_catalog import get_story_fixture


def _quality(*, syllables: int = 55) -> dict[str, Any]:
    return {
        "status": "PASS",
        "analysisStatus": "FULL",
        "contractPass": True,
        "contractFailures": [],
        "writtenSyllableCount": syllables,
        "directDialogueCount": 1,
        "excludedOverageCount": 0,
        "limitedOverageCount": 0,
        "riskPer10": 0.0,
        "perSkill": [],
    }


def _visual_scene(
    runtime: dict[str, Any],
    *,
    page_number: int,
) -> dict[str, Any]:
    characters = []
    for index, character in enumerate(runtime["storyState"]["characters"]):
        characters.append(
            {
                "characterId": character["characterId"],
                "present": True,
                "position": f"장면 영역 {index + 1}번 위치",
                "orientation": "이동 방향",
                "gazeTarget": "다음 길",
                "action": "다음 사건을 향해 걸어가요.",
                "emotion": {
                    "type": "CURIOUS",
                    "intensity": "LOW",
                },
            }
        )
    return {
        "shot": "WIDE_ESTABLISHING",
        "characters": characters,
        "mustInclude": [f"{page_number}페이지의 다음 길"],
        "mustAvoid": ["글자와 말풍선", "같은 캐릭터 중복"],
    }


def _chapter_response(
    runtime: dict[str, Any],
    *,
    chapter_number: int,
    page_count: int,
    generation_id: str,
    branching: bool,
) -> dict[str, Any]:
    question = "주인공은 다음에 어떤 말을 할까요?" if branching else None
    pages = []
    page_qualities = []
    for page_number in range(1, page_count + 1):
        is_last = page_number == page_count
        page_question = question if is_last else None
        pages.append(
            {
                "pageNumber": page_number,
                "sentences": [
                    f"주인공은 {page_number}쪽의 새 길을 천천히 걸어요.",
                    "친구는 작은 소리를 듣고 가까이 다가와요.",
                    "“우리 함께 가자.” 주인공이 다정히 말해요.",
                    "두 친구는 다음 사건을 향해 나란히 움직여요.",
                ],
                "visualScene": _visual_scene(
                    runtime,
                    page_number=page_number,
                ),
                "question": page_question,
                "choices": (
                    ["용기를 내자고 말해요.", "함께 가자고 말해요."]
                    if page_question
                    else []
                ),
                "requiresBranchInput": bool(page_question),
            }
        )
        page_qualities.append(
            {
                "pageNumber": page_number,
                "quality": _quality(),
            }
        )
    profile = runtime["generationProfile"]
    previous_question = runtime["storyState"]["lastQuestion"]
    return {
        "requestId": f"chapter-response-{chapter_number}",
        "schemaVersion": 3,
        "generationId": generation_id,
        "storyId": runtime["storyId"],
        "storyRevision": runtime["storyRevision"],
        "chapterNumber": chapter_number,
        "pages": pages,
        "quality": {
            "chapter": {
                **_quality(syllables=55 * page_count),
                "directDialogueCount": page_count,
            },
            "pages": page_qualities,
        },
        "generation": {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "promptVersion": "dynamic-chapter-test",
            "generationProfileVersion": profile[
                "generationProfileVersion"
            ],
            "policyHash": profile["policyHash"],
            "candidateCount": 1,
            "selectedCandidateId": "candidate-1",
            "pageCount": page_count,
            "apiCallCount": 1,
            "repairAttempted": False,
            "repairAccepted": False,
            "changedSentences": [],
            "repairDecisionReasons": [],
            "visualSceneStatus": "LLM_GENERATED",
            "visualSceneModel": "gpt-5.4-mini",
            "visualScenePromptVersion": "visual-scene-test",
            "visualSceneFallbackReason": None,
        },
        "timingMs": {
            "generation": 2200.0,
            "analysis": 180.0,
            "pagination": 20.0,
            "repair": 0.0,
            "visualScene": 100.0,
            "total": 2450.0,
        },
        "statePatch": {
            "expectedBaseRevision": runtime["storyRevision"],
            "rollingSummary": (
                f"{chapter_number}장의 사건이 차례대로 이어졌어요."
            ),
            "resolvedFactsAdded": [
                f"{chapter_number}장의 핵심 사건이 일어났어요."
            ],
            "unresolvedHooksAdded": [question] if question else [],
            "unresolvedHooksRemoved": (
                [previous_question] if previous_question else []
            ),
            "charactersUpserted": [],
            "lastQuestion": question,
        },
    }


def test_build_chapter_request_uses_v3_ordered_events_and_profile() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(
        story,
        profile_key="beginner",
    )

    payload = build_chapter_request(
        story,
        runtime,
        0,
        None,
        "dynamic-first-chapter",
    )

    StoryChapterGenerateRequest.model_validate(payload)
    assert payload["schemaVersion"] == 3
    assert payload["chapterNumber"] == 1
    assert payload["storyRevision"] == 0
    assert payload["branchInput"] is None
    assert payload["chapterPlan"]["minPages"] == 2
    assert payload["chapterPlan"]["maxPages"] == 4
    assert len(payload["chapterPlan"]["orderedEvents"]) == 4
    assert [
        event["lockedEvent"]
        for event in payload["chapterPlan"]["orderedEvents"]
    ] == [page.locked_event for page in story.beats[0].pages]
    assert payload["generationProfile"]["generationProfileVersion"] == 5
    assert payload["generationProfile"]["skills"]


def test_next_chapter_requires_branch_and_saved_last_question() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    first_response = _chapter_response(
        runtime,
        chapter_number=1,
        page_count=2,
        generation_id="first-branching-chapter",
        branching=True,
    )
    runtime = apply_chapter_response(runtime, first_response)

    with pytest.raises(DynamicStoryStateError, match="아이 답"):
        build_chapter_request(
            story,
            runtime,
            1,
            None,
            "missing-branch-input",
        )

    payload = build_chapter_request(
        story,
        runtime,
        1,
        {"source": "CHOICE", "text": "함께 가자고 말해요."},
        "second-chapter-with-branch",
    )

    assert payload["chapterNumber"] == 2
    assert payload["branchInput"]["source"] == "CHOICE"
    assert payload["branchInput"]["text"] == "함께 가자고 말해요."
    assert payload["storyState"]["lastQuestion"] == (
        first_response["pages"][-1]["question"]
    )


def test_first_chapter_rejects_branch_input() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)

    with pytest.raises(DynamicStoryStateError, match="첫 장"):
        build_chapter_request(
            story,
            runtime,
            0,
            {"source": "CHOICE", "text": "먼저 가요."},
            "invalid-first-branch",
        )


@pytest.mark.parametrize("page_count", [2, 3, 4])
def test_apply_chapter_adds_all_pages_but_increments_revision_once(
    page_count: int,
) -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    original = copy.deepcopy(runtime)
    response = _chapter_response(
        runtime,
        chapter_number=1,
        page_count=page_count,
        generation_id=f"dynamic-{page_count}-pages",
        branching=True,
    )

    updated = apply_chapter_response(runtime, response)

    assert runtime == original
    assert updated["storyRevision"] == 1
    assert updated["lastAppliedChapterNumber"] == 1
    assert len(updated["storyState"]["recentPages"]) == page_count
    assert updated["appliedGenerationIds"] == [
        f"dynamic-{page_count}-pages"
    ]
    assert updated["storyState"]["lastQuestion"] == (
        response["pages"][-1]["question"]
    )


def test_apply_rejects_duplicate_generation_id_without_mutation() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    response = _chapter_response(
        runtime,
        chapter_number=1,
        page_count=2,
        generation_id="duplicate-chapter-generation",
        branching=True,
    )
    updated = apply_chapter_response(runtime, response)
    snapshot = copy.deepcopy(updated)

    with pytest.raises(DynamicStoryStateError, match="이미 적용"):
        apply_chapter_response(updated, response)

    assert updated == snapshot


def test_recent_pages_are_capped_after_whole_chapter_patches() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)

    for chapter_number in range(1, 4):
        response = _chapter_response(
            runtime,
            chapter_number=chapter_number,
            page_count=4,
            generation_id=f"capped-chapter-{chapter_number}",
            branching=True,
        )
        runtime = apply_chapter_response(runtime, response)

    assert runtime["storyRevision"] == 3
    assert runtime["lastAppliedChapterNumber"] == 3
    assert len(runtime["storyState"]["recentPages"]) == 8


def test_question_is_allowed_only_on_last_page() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    response = _chapter_response(
        runtime,
        chapter_number=1,
        page_count=3,
        generation_id="invalid-early-question",
        branching=True,
    )
    response["pages"][0]["question"] = "너무 이른 질문인가요?"
    response["pages"][0]["choices"] = ["네.", "아니요."]
    response["pages"][0]["requiresBranchInput"] = True

    with pytest.raises(DynamicStoryStateError):
        apply_chapter_response(runtime, response)


def test_display_and_generation_cursors_advance_independently() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    response = _chapter_response(
        runtime,
        chapter_number=1,
        page_count=3,
        generation_id="separate-cursors",
        branching=True,
    )
    generation_cursor = generation_cursor_after_response(
        story,
        initial_generation_cursor(),
        response,
    )
    display_cursor = initial_display_cursor(response)

    assert generation_cursor == ChapterGenerationCursor(
        chapter_index=1,
        awaiting_branch=True,
    )
    assert displayed_page(response, display_cursor)["pageNumber"] == 1
    assert display_cursor.is_last_page is False

    display_cursor = advance_display_cursor(display_cursor)
    assert displayed_page(response, display_cursor)["pageNumber"] == 2
    assert generation_cursor.chapter_index == 1
    assert generation_cursor.awaiting_branch is True

    display_cursor = advance_display_cursor(display_cursor)
    final_page = displayed_page(response, display_cursor)
    assert display_cursor.is_last_page is True
    assert final_page["requiresBranchInput"] is True
    assert final_page["question"]

    with pytest.raises(DynamicStoryStateError, match="마지막 페이지"):
        advance_display_cursor(display_cursor)


def test_final_chapter_marks_generation_complete() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    runtime["storyRevision"] = story.total_chapters - 1
    runtime["lastAppliedChapterNumber"] = story.total_chapters - 1
    runtime["storyState"]["lastQuestion"] = "마지막 장으로 갈까요?"
    response = _chapter_response(
        runtime,
        chapter_number=story.total_chapters,
        page_count=2,
        generation_id="final-dynamic-chapter",
        branching=False,
    )
    cursor = ChapterGenerationCursor(
        chapter_index=story.total_chapters - 1,
        awaiting_branch=True,
    )

    updated = apply_chapter_response(runtime, response)
    next_cursor = generation_cursor_after_response(
        story,
        cursor,
        response,
    )

    assert updated["storyRevision"] == story.total_chapters
    assert updated["storyState"]["lastQuestion"] is None
    assert next_cursor.complete is True
    assert next_cursor.awaiting_branch is False


def test_final_chapter_request_has_no_question_focus() -> None:
    story = get_story_fixture(1001)
    runtime = initial_dynamic_runtime(story)
    runtime["storyRevision"] = story.total_chapters - 1
    runtime["lastAppliedChapterNumber"] = story.total_chapters - 1
    runtime["storyState"]["lastQuestion"] = "마지막 장으로 갈까요?"

    payload = build_chapter_request(
        story,
        runtime,
        story.total_chapters - 1,
        {"source": "TEXT_CONFIRMED", "text": "함께 웃으며 끝내요."},
        "dynamic-final-chapter",
    )

    assert payload["conclude"] is True
    assert payload["chapterPlan"]["questionFocus"] is None
