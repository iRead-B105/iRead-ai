from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.devtools.reading_profiles import (
    READING_PROFILE_PRESETS,
    build_generation_profile,
)
from iread_ai.devtools.service_story_catalog import ServiceStoryFixture


class DynamicStoryStateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChapterGenerationCursor:
    chapter_index: int = 0
    awaiting_branch: bool = False
    complete: bool = False


@dataclass(frozen=True, slots=True)
class ChapterDisplayCursor:
    chapter_number: int
    page_index: int
    page_count: int

    @property
    def page_number(self) -> int:
        return self.page_index + 1

    @property
    def is_last_page(self) -> bool:
        return self.page_index == self.page_count - 1


def initial_dynamic_runtime(
    story: ServiceStoryFixture,
    *,
    profile_key: str = "balanced",
    student_id: int = 7,
) -> dict[str, Any]:
    if profile_key not in READING_PROFILE_PRESETS:
        raise KeyError(f"Unknown reading profile preset: {profile_key}")
    return {
        "storyId": story.template_id,
        "studentId": student_id,
        "totalChapters": story.total_chapters,
        "lastAppliedChapterNumber": 0,
        "storyRevision": 0,
        "storyState": {
            "rollingSummary": story.initial_summary,
            "resolvedFacts": [],
            "unresolvedHooks": [],
            "recentPages": [],
            "characters": [
                {
                    "characterId": character.character_id,
                    "name": character.name,
                    "role": character.role,
                    "immutableTraits": list(character.immutable_traits),
                }
                for character in story.characters
            ],
            "lastQuestion": None,
        },
        "generationProfile": build_generation_profile(story, profile_key),
        "appliedGenerationIds": [],
    }


def initial_generation_cursor() -> ChapterGenerationCursor:
    return ChapterGenerationCursor()


def build_chapter_request(
    story: ServiceStoryFixture,
    runtime: Mapping[str, Any],
    chapter_index: int,
    branch_input: Mapping[str, str] | None,
    request_id: str,
) -> dict[str, Any]:
    if chapter_index < 0 or chapter_index >= story.total_chapters:
        raise DynamicStoryStateError("생성할 이야기 장이 범위를 벗어났습니다.")
    expected_index = int(runtime.get("lastAppliedChapterNumber", 0))
    if chapter_index != expected_index:
        raise DynamicStoryStateError("요청 장과 마지막으로 저장한 이야기 장이 이어지지 않습니다.")
    if int(runtime.get("storyId", 0)) != story.template_id:
        raise DynamicStoryStateError("선택한 이야기와 runtime의 storyId가 다릅니다.")

    if chapter_index == 0:
        if branch_input is not None:
            raise DynamicStoryStateError("첫 장에는 이전 질문의 아이 답을 전달할 수 없습니다.")
    else:
        if branch_input is None:
            raise DynamicStoryStateError("다음 장을 생성하려면 직전 질문의 아이 답이 필요합니다.")
        last_question = _story_state(runtime).get("lastQuestion")
        if not isinstance(last_question, str) or not last_question.strip():
            raise DynamicStoryStateError(
                "직전 장의 질문이 저장되지 않아 다음 장을 생성할 수 없습니다."
            )
        _validate_branch_input(branch_input)

    beat = story.beats[chapter_index]
    payload: dict[str, Any] = {
        "requestId": request_id,
        "schemaVersion": 3,
        "storyId": story.template_id,
        "studentId": int(runtime.get("studentId", 7)),
        "storyRevision": int(runtime["storyRevision"]),
        "chapterNumber": chapter_index + 1,
        "conclude": beat.concluding,
        "storyTemplate": {
            "templateId": story.template_id,
            "version": story.version,
            "title": story.title,
            "context": story.context,
            "currentBeat": {
                "beatId": beat.beat_id,
                "goal": beat.goal,
                "questionFocus": beat.question_focus,
                "allowedBranchSlots": list(beat.allowed_branch_slots),
            },
        },
        "storyState": copy.deepcopy(_story_state(runtime)),
        "chapterPlan": {
            "orderedEvents": [
                {
                    "eventId": f"{beat.beat_id}-event-{index}",
                    "lockedEvent": page.locked_event,
                    "requiredCharacters": list(page.required_character_ids),
                    "requiredConcepts": list(page.required_concepts),
                }
                for index, page in enumerate(beat.pages, start=1)
            ],
            "minPages": 2,
            "maxPages": 4,
            "questionFocus": (None if beat.concluding else beat.question_focus),
        },
        "branchInput": (dict(branch_input) if branch_input is not None else None),
        "generationProfile": copy.deepcopy(_generation_profile(runtime)),
    }
    try:
        request = StoryChapterGenerateRequest.model_validate(payload)
    except ValueError as exc:
        raise DynamicStoryStateError(str(exc)) from exc
    return request.model_dump(by_alias=True)


def apply_chapter_response(
    runtime: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        response = StoryChapterGenerateResponse.model_validate(response_payload)
    except ValueError as exc:
        raise DynamicStoryStateError(str(exc)) from exc

    current_revision = int(runtime["storyRevision"])
    current_chapter = int(runtime.get("lastAppliedChapterNumber", 0))
    applied_ids = [str(value) for value in runtime.get("appliedGenerationIds", [])]
    if response.generation_id in applied_ids:
        raise DynamicStoryStateError("이미 적용한 generationId입니다.")
    if response.story_id != int(runtime.get("storyId", 0)):
        raise DynamicStoryStateError("응답 storyId가 현재 이야기와 다릅니다.")
    if response.story_revision != current_revision:
        raise DynamicStoryStateError("응답 storyRevision이 현재 이야기 revision과 다릅니다.")
    if response.state_patch.expected_base_revision != current_revision:
        raise DynamicStoryStateError("응답의 expectedBaseRevision이 현재 revision과 다릅니다.")
    if response.chapter_number != current_chapter + 1:
        raise DynamicStoryStateError("응답 장 번호가 마지막으로 저장한 장과 이어지지 않습니다.")
    if response.generation.policy_hash != str(_generation_profile(runtime).get("policyHash")):
        raise DynamicStoryStateError("응답 policyHash가 요청에 사용한 프로필과 다릅니다.")
    if response.generation.generation_profile_version != int(
        _generation_profile(runtime).get(
            "generationProfileVersion",
            0,
        )
    ):
        raise DynamicStoryStateError("응답 generationProfileVersion이 현재 프로필과 다릅니다.")

    total_chapters = int(runtime.get("totalChapters", 0))
    is_final_chapter = response.chapter_number == total_chapters
    final_page = response.pages[-1]
    if final_page.requires_branch_input == is_final_chapter:
        raise DynamicStoryStateError("마지막 장 여부와 최종 페이지의 분기 상태가 맞지 않습니다.")
    if response.state_patch.last_question != final_page.question:
        raise DynamicStoryStateError("statePatch.lastQuestion이 마지막 페이지 질문과 다릅니다.")

    updated = copy.deepcopy(dict(runtime))
    story_state = copy.deepcopy(_story_state(updated))
    patch = response.state_patch
    story_state["rollingSummary"] = patch.rolling_summary
    story_state["resolvedFacts"] = _ordered_unique(
        [
            *story_state.get("resolvedFacts", []),
            *patch.resolved_facts_added,
        ]
    )
    removed_hooks = set(patch.unresolved_hooks_removed)
    remaining_hooks = [
        hook for hook in story_state.get("unresolvedHooks", []) if hook not in removed_hooks
    ]
    story_state["unresolvedHooks"] = _ordered_unique(
        [*remaining_hooks, *patch.unresolved_hooks_added]
    )

    characters = {
        str(character["characterId"]): copy.deepcopy(character)
        for character in story_state.get("characters", [])
    }
    for character in patch.characters_upserted:
        document = character.model_dump(by_alias=True)
        characters[str(document["characterId"])] = document
    story_state["characters"] = list(characters.values())
    story_state["lastQuestion"] = patch.last_question

    recent_pages = list(story_state.get("recentPages", []))
    recent_pages.extend(
        {
            "pageNumber": page.page_number,
            "sentences": list(page.sentences),
            "question": page.question,
        }
        for page in response.pages
    )
    story_state["recentPages"] = recent_pages[-8:]

    updated["storyState"] = story_state
    updated["storyRevision"] = current_revision + 1
    updated["lastAppliedChapterNumber"] = response.chapter_number
    updated["appliedGenerationIds"] = [
        *applied_ids,
        response.generation_id,
    ]
    return updated


def generation_cursor_after_response(
    story: ServiceStoryFixture,
    cursor: ChapterGenerationCursor,
    response_payload: Mapping[str, Any],
) -> ChapterGenerationCursor:
    if cursor.complete:
        raise DynamicStoryStateError("이미 완결된 이야기입니다.")
    try:
        response = StoryChapterGenerateResponse.model_validate(response_payload)
    except ValueError as exc:
        raise DynamicStoryStateError(str(exc)) from exc
    if response.chapter_number != cursor.chapter_index + 1:
        raise DynamicStoryStateError("응답 장 번호와 생성 커서가 다릅니다.")

    final_page = response.pages[-1]
    if final_page.requires_branch_input:
        next_index = cursor.chapter_index + 1
        if next_index >= story.total_chapters:
            raise DynamicStoryStateError("분기 질문 뒤에 생성할 다음 장 계획이 없습니다.")
        return ChapterGenerationCursor(
            chapter_index=next_index,
            awaiting_branch=True,
        )
    if cursor.chapter_index != story.total_chapters - 1:
        raise DynamicStoryStateError("완결되지 않은 이야기인데 다음 분기 질문이 없습니다.")
    return ChapterGenerationCursor(
        chapter_index=cursor.chapter_index,
        complete=True,
    )


def initial_display_cursor(
    response_payload: Mapping[str, Any],
) -> ChapterDisplayCursor:
    try:
        response = StoryChapterGenerateResponse.model_validate(response_payload)
    except ValueError as exc:
        raise DynamicStoryStateError(str(exc)) from exc
    return ChapterDisplayCursor(
        chapter_number=response.chapter_number,
        page_index=0,
        page_count=len(response.pages),
    )


def advance_display_cursor(
    cursor: ChapterDisplayCursor,
) -> ChapterDisplayCursor:
    if cursor.is_last_page:
        raise DynamicStoryStateError("이미 이 장의 마지막 페이지입니다.")
    return ChapterDisplayCursor(
        chapter_number=cursor.chapter_number,
        page_index=cursor.page_index + 1,
        page_count=cursor.page_count,
    )


def displayed_page(
    response_payload: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
) -> dict[str, Any]:
    try:
        response = StoryChapterGenerateResponse.model_validate(response_payload)
    except ValueError as exc:
        raise DynamicStoryStateError(str(exc)) from exc
    if response.chapter_number != cursor.chapter_number:
        raise DynamicStoryStateError("화면 커서의 장 번호가 응답과 다릅니다.")
    if len(response.pages) != cursor.page_count:
        raise DynamicStoryStateError("화면 커서의 pageCount가 응답과 다릅니다.")
    if cursor.page_index < 0 or cursor.page_index >= cursor.page_count:
        raise DynamicStoryStateError("화면 페이지 위치가 범위를 벗어났습니다.")
    return response.pages[cursor.page_index].model_dump(by_alias=True)


def _story_state(runtime: Mapping[str, Any]) -> dict[str, Any]:
    value = runtime.get("storyState")
    if not isinstance(value, Mapping):
        raise DynamicStoryStateError("runtime.storyState가 없습니다.")
    return {str(key): item for key, item in value.items()}


def _generation_profile(runtime: Mapping[str, Any]) -> dict[str, Any]:
    value = runtime.get("generationProfile")
    if not isinstance(value, Mapping):
        raise DynamicStoryStateError("runtime.generationProfile이 없습니다.")
    return {str(key): item for key, item in value.items()}


def _validate_branch_input(branch_input: Mapping[str, str]) -> None:
    source = str(branch_input.get("source", "")).strip()
    text = str(branch_input.get("text", "")).strip()
    if source not in {"CHOICE", "TEXT_CONFIRMED", "STT_CONFIRMED"}:
        raise DynamicStoryStateError("아이 답 source가 올바르지 않습니다.")
    if not text:
        raise DynamicStoryStateError("아이 답 text가 비어 있습니다.")


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "ChapterDisplayCursor",
    "ChapterGenerationCursor",
    "DynamicStoryStateError",
    "advance_display_cursor",
    "apply_chapter_response",
    "build_chapter_request",
    "displayed_page",
    "generation_cursor_after_response",
    "initial_display_cursor",
    "initial_dynamic_runtime",
    "initial_generation_cursor",
]
