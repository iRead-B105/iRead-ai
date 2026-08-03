from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)

_POLICY_HASH = "sha256:" + "a" * 64


def request_payload(
    *,
    conclude: bool = False,
    min_pages: int = 2,
    max_pages: int = 4,
) -> dict[str, Any]:
    question_focus = None if conclude else "토끼가 다음에 고를 재미있는 행동"
    return {
        "requestId": "req-story-101-chapter-2",
        "schemaVersion": 3,
        "storyId": 101,
        "studentId": 7,
        "storyRevision": 8,
        "chapterNumber": 2,
        "conclude": conclude,
        "storyTemplate": {
            "templateId": 11,
            "version": 3,
            "title": "토끼와 거북이",
            "context": "토끼와 거북이가 숲길에서 경주해요.",
            "currentBeat": {
                "beatId": "hare-rests",
                "goal": "토끼가 쉬는 동안 거북이가 계속 나아가요.",
                "questionFocus": question_focus,
                "allowedBranchSlots": ["FUNNY_SOUND", "HELPFUL_ITEM"],
            },
        },
        "storyState": {
            "rollingSummary": "경주가 시작되었고 토끼가 먼저 달려갔어요.",
            "resolvedFacts": ["경주가 시작되었어요."],
            "unresolvedHooks": ["토끼와 거북이가 언덕을 넘어야 해요."],
            "recentPages": [
                {
                    "pageNumber": 3,
                    "sentences": [
                        "거북이는 높은 언덕 앞에 섰어요.",
                        "토끼는 나무 아래에서 쉬었어요.",
                        "“나는 계속 갈 거야.” 거북이가 말했어요.",
                    ],
                    "question": "어떤 소리로 경주를 시작할까요?",
                }
            ],
            "characters": [
                {
                    "characterId": "hare",
                    "name": "토끼",
                    "role": "주인공",
                    "immutableTraits": ["빠르다", "처음에는 자만한다"],
                },
                {
                    "characterId": "tortoise",
                    "name": "거북이",
                    "role": "주인공",
                    "immutableTraits": ["천천히 간다", "포기하지 않는다"],
                },
            ],
            "lastQuestion": "어떤 소리로 경주를 시작할까요?",
        },
        "chapterPlan": {
            "orderedEvents": [
                {
                    "eventId": "child-answer-event",
                    "lockedEvent": "아이의 답에 담긴 소리가 숲에 실제로 울려요.",
                    "requiredCharacters": ["hare", "tortoise"],
                    "requiredConcepts": ["선택한 소리", "두 주인공의 반응"],
                },
                {
                    "eventId": "return-to-race",
                    "lockedEvent": "짧은 소동 뒤 두 주인공이 경주로 돌아가요.",
                    "requiredCharacters": ["hare", "tortoise"],
                    "requiredConcepts": ["경주 재개"],
                },
            ],
            "minPages": min_pages,
            "maxPages": max_pages,
            "questionFocus": question_focus,
        },
        "branchInput": {
            "source": "TEXT_CONFIRMED",
            "text": "방구 소리로 출발해요!",
        },
        "generationProfile": {
            "schemaVersion": 2,
            "generationProfileVersion": 7,
            "sourceReadingProfileVersion": 12,
            "compilerVersion": "reading-policy-v2",
            "policyHash": _POLICY_HASH,
            "contentContract": {
                "sentenceCount": 4,
                "preferredWrittenSyllables": {"min": 55, "max": 70},
                "acceptedWrittenSyllables": {"min": 50, "max": 75},
                "directDialogueCount": 1,
            },
            "skills": [
                {
                    "code": "ONSET_ㄲ",
                    "role": "LIMITED",
                    "maxOccurrences": 1,
                    "targetMin": None,
                    "targetMax": None,
                    "unitPenalty": 1.2,
                }
            ],
            "protectedTerms": ["토끼", "거북이"],
        },
    }


def _quality(
    *,
    written_syllables: int,
    dialogue_count: int,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "analysisStatus": "FULL",
        "contractPass": True,
        "contractFailures": [],
        "writtenSyllableCount": written_syllables,
        "directDialogueCount": dialogue_count,
        "excludedOverageCount": 0,
        "limitedOverageCount": 0,
        "riskPer10": 0.2,
        "perSkill": [],
    }


def _visual_scene(page_number: int) -> dict[str, Any]:
    return {
        "shot": "WIDE_THREE_QUARTER",
        "characters": [
            {
                "characterId": "hare",
                "present": True,
                "position": "화면 왼쪽",
                "orientation": "이동 방향",
                "gazeTarget": "결승선",
                "action": "토끼가 숲길을 달려가요.",
                "emotion": {
                    "type": "EXCITED",
                    "intensity": "MEDIUM",
                },
            },
            {
                "characterId": "tortoise",
                "present": True,
                "position": "화면 오른쪽",
                "orientation": "이동 방향",
                "gazeTarget": "결승선",
                "action": "거북이가 천천히 걸어가요.",
                "emotion": {
                    "type": "FOCUSED",
                    "intensity": "LOW",
                },
            },
        ],
        "mustInclude": [f"{page_number}페이지 숲길", "토끼", "거북이"],
        "mustAvoid": ["글자와 말풍선", "같은 캐릭터 중복"],
    }


def response_payload(
    *,
    page_count: int = 3,
    conclude: bool = False,
) -> dict[str, Any]:
    page_sentences = [
        [
            "숲에 방구 소리가 우렁차게 울렸어요.",
            "토끼는 깜짝 놀라 귀를 쫑긋 세웠어요.",
            "“누가 북을 친 거야?” 토끼가 물었어요.",
        ],
        [
            "거북이는 웃음을 참고 한 걸음 나아갔어요.",
            "토끼도 장난을 멈추고 다시 달리기 시작했어요.",
            "두 친구의 발자국이 언덕길에 이어졌어요.",
            "작은 새들이 머리 위에서 응원했어요.",
        ],
        [
            "언덕 너머에서 반짝이는 깃발이 보였어요.",
            "거북이는 천천히 숨을 고르고 길을 살폈어요.",
            "“이제 어느 길로 갈까?” 토끼가 물었어요.",
        ],
        [
            "둘은 좁은 꽃길 앞에서 잠시 멈췄어요.",
            "거북이는 바람 냄새를 맡으며 고개를 끄덕였어요.",
            "토끼는 먼저 달리지 않고 친구를 기다렸어요.",
        ],
    ][:page_count]
    written_syllables = [52, 61, 54, 56][:page_count]
    dialogue_counts = [1, 0, 1, 0][:page_count]
    pages: list[dict[str, Any]] = []
    for index, sentences in enumerate(page_sentences, start=1):
        final_page = index == page_count
        branching = final_page and not conclude
        pages.append(
            {
                "pageNumber": index,
                "sentences": sentences,
                "visualScene": _visual_scene(index),
                "question": (
                    "두 친구는 어느 길로 달려갈까요?"
                    if branching
                    else None
                ),
                "subtitle": "숲길의 새로운 갈림길" if branching else None,
                "choices": (
                    ["꽃길로 가요.", "냇물 길로 가요.", "숲길로 가요."]
                    if branching
                    else []
                ),
                "requiresBranchInput": branching,
            }
        )

    page_qualities = [
        {
            "pageNumber": index,
            "quality": _quality(
                written_syllables=written_syllables[index - 1],
                dialogue_count=dialogue_counts[index - 1],
            ),
        }
        for index in range(1, page_count + 1)
    ]
    return {
        "requestId": "req-story-101-chapter-2",
        "schemaVersion": 3,
        "generationId": "chapter-8f2b",
        "storyId": 101,
        "storyRevision": 8,
        "chapterNumber": 2,
        "pages": pages,
        "quality": {
            "chapter": _quality(
                written_syllables=sum(written_syllables),
                dialogue_count=sum(dialogue_counts),
            ),
            "pages": page_qualities,
        },
        "generation": {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "promptVersion": "chapter-v3.1",
            "generationProfileVersion": 7,
            "policyHash": _POLICY_HASH,
            "candidateCount": 3,
            "selectedCandidateId": "candidate-2",
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
            "generation": 3200.0,
            "analysis": 210.0,
            "pagination": 1.2,
            "repair": 0.0,
            "visualScene": 120.0,
            "total": 3411.2,
        },
        "statePatch": {
            "expectedBaseRevision": 8,
            "rollingSummary": (
                "방구 소리로 소동이 생겼지만 토끼와 거북이는 다시 경주해요."
            ),
            "resolvedFactsAdded": ["두 친구가 소동 뒤 경주로 돌아갔어요."],
            "unresolvedHooksAdded": (
                ["두 친구는 어느 길로 달려갈까요?"]
                if not conclude
                else []
            ),
            "unresolvedHooksRemoved": [
                "어떤 소리로 경주를 시작할까요?"
            ],
            "charactersUpserted": [],
            "lastQuestion": (
                "두 친구는 어느 길로 달려갈까요?"
                if not conclude
                else None
            ),
        },
    }


def test_chapter_request_round_trips_strict_camel_case() -> None:
    payload = request_payload()

    request = StoryChapterGenerateRequest.model_validate(payload)

    assert request.schema_version == 3
    assert request.generation_profile.schema_version == 2
    assert request.chapter_plan.min_pages == 2
    assert request.chapter_plan.ordered_events[1].event_id == "return-to-race"
    assert request.model_dump(by_alias=True) == payload


def test_dynamic_chapter_response_round_trips() -> None:
    payload = response_payload(page_count=3)

    response = StoryChapterGenerateResponse.model_validate(payload)

    assert len(response.pages) == 3
    assert len(response.pages[1].sentences) == 4
    assert response.pages[-1].requires_branch_input is True
    assert response.generation.page_count == 3
    assert response.pages[0].visual_scene.characters[0].emotion.type == "EXCITED"
    assert response.model_dump(by_alias=True) == payload


def test_visual_scene_separates_absence_from_emotion() -> None:
    payload = response_payload(page_count=2)
    absent = payload["pages"][0]["visualScene"]["characters"][0]
    absent.update(
        {
            "present": False,
            "position": None,
            "orientation": None,
            "gazeTarget": None,
            "action": None,
            "emotion": None,
        }
    )

    response = StoryChapterGenerateResponse.model_validate(payload)

    assert response.pages[0].visual_scene.characters[0].present is False
    assert response.pages[0].visual_scene.characters[0].emotion is None

    absent["emotion"] = {"type": "CALM", "intensity": "LOW"}
    with pytest.raises(ValidationError, match="must not define"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_visual_scene_requires_details_for_present_character() -> None:
    payload = response_payload(page_count=2)
    payload["pages"][0]["visualScene"]["characters"][0]["action"] = None

    with pytest.raises(ValidationError, match="requires position"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_response_rejects_a_five_sentence_page() -> None:
    payload = response_payload(page_count=2)
    payload["pages"][0]["sentences"].extend(
        ["다섯 번째 문장은 허용하지 않아요.", "여섯 번째 문장도 쓰지 않아요."]
    )

    with pytest.raises(ValidationError, match="at most 4"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_request_rejects_unknown_character_and_duplicate_event_ids() -> None:
    payload = request_payload()
    payload["chapterPlan"]["orderedEvents"][0]["requiredCharacters"] = [
        "unknown"
    ]

    with pytest.raises(ValidationError, match="unknown characterId"):
        StoryChapterGenerateRequest.model_validate(payload)

    payload = request_payload()
    payload["chapterPlan"]["orderedEvents"][1]["eventId"] = (
        payload["chapterPlan"]["orderedEvents"][0]["eventId"]
    )

    with pytest.raises(ValidationError, match="eventId values must be unique"):
        StoryChapterGenerateRequest.model_validate(payload)


def test_request_enforces_page_bounds_and_question_focus() -> None:
    payload = request_payload()
    payload["chapterPlan"]["minPages"] = 4
    payload["chapterPlan"]["maxPages"] = 2

    with pytest.raises(ValidationError, match="minPages"):
        StoryChapterGenerateRequest.model_validate(payload)

    payload = request_payload()
    payload["chapterPlan"]["questionFocus"] = None

    with pytest.raises(ValidationError, match="required"):
        StoryChapterGenerateRequest.model_validate(payload)

    payload = request_payload(conclude=True)
    payload["chapterPlan"]["questionFocus"] = "끝인데 질문하면 안 돼요."

    with pytest.raises(ValidationError, match="not allowed"):
        StoryChapterGenerateRequest.model_validate(payload)


def test_response_requires_ordered_pages_and_matching_quality_pages() -> None:
    payload = response_payload()
    payload["pages"][1]["pageNumber"] = 3

    with pytest.raises(ValidationError, match="ordered"):
        StoryChapterGenerateResponse.model_validate(payload)

    payload = response_payload()
    removed = payload["quality"]["pages"].pop()["quality"]
    chapter_quality = payload["quality"]["chapter"]
    for field_name in (
        "writtenSyllableCount",
        "directDialogueCount",
        "excludedOverageCount",
        "limitedOverageCount",
    ):
        chapter_quality[field_name] -= removed[field_name]

    with pytest.raises(ValidationError, match="one entry"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_response_allows_branching_only_on_final_page() -> None:
    payload = response_payload()
    payload["pages"][0].update(
        {
            "subtitle": "먼저 만난 갈림길",
            "question": "먼저 고를까요?",
            "choices": ["네.", "아니요.", "아직 모르겠어요."],
            "requiresBranchInput": True,
        }
    )

    with pytest.raises(ValidationError, match="only on the final page"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_response_validates_chapter_quality_totals() -> None:
    payload = response_payload()
    payload["quality"]["chapter"]["writtenSyllableCount"] += 1

    with pytest.raises(ValidationError, match="sum of page qualities"):
        StoryChapterGenerateResponse.model_validate(payload)


def test_response_validates_changed_sentence_coordinates() -> None:
    payload = response_payload()
    payload["generation"].update(
        {
            "repairAttempted": True,
            "repairAccepted": True,
            "changedSentences": [
                {
                    "globalSentenceNumber": 5,
                    "pageNumber": 2,
                    "sentenceNumber": 1,
                }
            ],
        }
    )

    with pytest.raises(ValidationError, match="globalSentenceNumber"):
        StoryChapterGenerateResponse.model_validate(payload)

    payload["generation"]["changedSentences"][0]["globalSentenceNumber"] = 4
    response = StoryChapterGenerateResponse.model_validate(payload)

    assert response.generation.changed_sentences[0].page_number == 2


@pytest.mark.parametrize("page_count", [2, 3, 4])
def test_response_cross_validation_accepts_each_dynamic_page_count(
    page_count: int,
) -> None:
    request = StoryChapterGenerateRequest.model_validate(
        request_payload(min_pages=2, max_pages=4)
    )
    response = StoryChapterGenerateResponse.model_validate(
        response_payload(page_count=page_count)
    )

    assert response.validate_against_request(request) is response


def test_response_cross_validation_enforces_requested_page_range() -> None:
    request = StoryChapterGenerateRequest.model_validate(
        request_payload(min_pages=2, max_pages=2)
    )
    response = StoryChapterGenerateResponse.model_validate(
        response_payload(page_count=3)
    )

    with pytest.raises(ValueError, match="page count"):
        response.validate_against_request(request)


def test_response_cross_validation_enforces_concluding_branch_mode() -> None:
    request = StoryChapterGenerateRequest.model_validate(
        request_payload(conclude=True)
    )
    response = StoryChapterGenerateResponse.model_validate(
        response_payload(conclude=False)
    )

    with pytest.raises(ValueError, match="non-concluding"):
        response.validate_against_request(request)

    concluding_response = StoryChapterGenerateResponse.model_validate(
        response_payload(conclude=True)
    )
    assert concluding_response.validate_against_request(request) is (
        concluding_response
    )


def test_response_cross_validation_rejects_snapshot_mismatch() -> None:
    request = StoryChapterGenerateRequest.model_validate(request_payload())
    payload = response_payload()
    payload["generation"]["policyHash"] = "sha256:" + "b" * 64
    response = StoryChapterGenerateResponse.model_validate(payload)

    with pytest.raises(ValueError, match="policyHash"):
        response.validate_against_request(request)


def test_response_cross_validation_rejects_visual_character_drift() -> None:
    request = StoryChapterGenerateRequest.model_validate(request_payload())
    payload = response_payload()
    characters = payload["pages"][0]["visualScene"]["characters"]
    characters.reverse()
    response = StoryChapterGenerateResponse.model_validate(payload)

    with pytest.raises(ValueError, match="request order"):
        response.validate_against_request(request)


def test_branch_input_is_optional_but_strict_when_present() -> None:
    payload = request_payload()
    payload.pop("branchInput")

    request = StoryChapterGenerateRequest.model_validate(payload)

    assert request.branch_input is None

    invalid = deepcopy(payload)
    invalid["branchInput"] = {
        "source": "RAW_STT",
        "text": "검증되지 않은 입력",
    }
    with pytest.raises(ValidationError):
        StoryChapterGenerateRequest.model_validate(invalid)
