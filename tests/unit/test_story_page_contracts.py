from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from iread_ai.contracts.story_page import (
    StoryPageGenerateRequest,
    StoryPageGenerateResponse,
)

_POLICY_HASH = "sha256:" + "a" * 64


def request_payload() -> dict[str, Any]:
    return {
        "requestId": "req-story-101-chapter-2-page-1",
        "schemaVersion": 2,
        "storyId": 101,
        "studentId": 7,
        "storyRevision": 8,
        "chapterNumber": 2,
        "conclude": False,
        "storyTemplate": {
            "templateId": 11,
            "version": 3,
            "title": "토끼와 거북이",
            "context": "토끼와 거북이가 숲길에서 경주해요.",
            "currentBeat": {
                "beatId": "hare-rests",
                "goal": "토끼가 쉬는 동안 거북이가 계속 나아가요.",
                "questionFocus": "거북이를 어떻게 응원할지",
                "allowedBranchSlots": ["CHEER_PHRASE", "HELPFUL_ITEM"],
            },
        },
        "storyState": {
            "rollingSummary": "경주가 시작되었고 토끼가 먼저 달려갔어요.",
            "resolvedFacts": ["경주가 시작되었어요."],
            "unresolvedHooks": ["거북이가 언덕을 넘어야 해요."],
            "recentPages": [
                {
                    "pageNumber": 4,
                    "sentences": [
                        "거북이는 높은 언덕 앞에 섰어요.",
                        "토끼는 나무 아래에서 쉬었어요.",
                        "“나는 계속 갈 거야.” 거북이가 말했어요.",
                        "거북이는 아이의 응원을 기다렸어요.",
                    ],
                    "question": "거북이를 어떻게 응원할까요?",
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
            "lastQuestion": "거북이를 어떻게 응원할까요?",
        },
        "pagePlan": {
            "pageNumber": 1,
            "lockedEvent": "아이의 응원을 들은 거북이가 언덕을 오르기 시작해요.",
            "requiredCharacters": ["tortoise"],
            "requiredConcepts": ["아이의 응원", "언덕을 오르기"],
            "questionFocus": None,
        },
        "branchInput": {
            "source": "STT_CONFIRMED",
            "text": "천천히 가도 괜찮아!",
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
                },
                {
                    "code": "PHONO_LIAISON",
                    "role": "EXCLUDED",
                    "maxOccurrences": 0,
                    "targetMin": None,
                    "targetMax": None,
                    "unitPenalty": 1.5,
                },
            ],
            "protectedTerms": ["토끼", "거북이"],
        },
    }


def response_payload(*, branching: bool = False) -> dict[str, Any]:
    question = "거북이는 다음에 무엇을 할까요?" if branching else None
    choices = ["언덕을 올라요.", "친구를 불러요."] if branching else []
    return {
        "requestId": "req-story-101-chapter-2-page-1",
        "schemaVersion": 2,
        "generationId": "gen-8f2b-page-1",
        "storyId": 101,
        "storyRevision": 8,
        "chapterNumber": 2,
        "page": {
            "pageNumber": 4 if branching else 1,
            "sentences": [
                "아이의 응원이 숲길에 또렷하게 울렸어요.",
                "거북이는 고개를 들고 언덕을 향해 걸었어요.",
                "“천천히 끝까지 갈게.” 거북이가 말했어요.",
                "작은 발자국이 언덕 위로 길게 이어졌어요.",
            ],
            "question": question,
            "choices": choices,
            "requiresBranchInput": branching,
        },
        "quality": {
            "status": "PASS",
            "analysisStatus": "FULL",
            "contractPass": True,
            "contractFailures": [],
            "writtenSyllableCount": 64,
            "directDialogueCount": 1,
            "excludedOverageCount": 0,
            "limitedOverageCount": 0,
            "riskPer10": 0.4,
            "perSkill": [
                {
                    "code": "ONSET_ㄲ",
                    "role": "LIMITED",
                    "status": "PASS",
                    "occurrences": 1,
                    "maxOccurrences": 1,
                    "targetMin": None,
                    "targetMax": None,
                    "overage": 0,
                    "targetDistance": None,
                    "weightedRisk": 0.0,
                }
            ],
        },
        "generation": {
            "provider": "openai",
            "model": "configured-story-model",
            "promptVersion": "page-v2.1",
            "generationProfileVersion": 7,
            "policyHash": _POLICY_HASH,
            "candidateCount": 1,
            "selectedCandidateId": "candidate-1",
            "apiCallCount": 2,
            "repairAttempted": True,
            "repairAccepted": False,
            "changedSentenceNumbers": [3],
            "repairDecisionReasons": ["SEMANTIC_OVERLAP_LOW:3"],
        },
        "timingMs": {
            "generation": 3100.0,
            "analysis": 168.0,
            "repair": 900.0,
            "total": 4168.0,
        },
        "statePatch": {
            "expectedBaseRevision": 8,
            "rollingSummary": (
                "경주가 시작되었고 아이의 응원을 들은 거북이가 언덕을 오르기 시작했어요."
            ),
            "resolvedFactsAdded": ["거북이가 아이의 응원을 들었어요."],
            "unresolvedHooksAdded": ["거북이가 언덕 꼭대기에 닿아야 해요."],
            "unresolvedHooksRemoved": [],
            "charactersUpserted": [],
            "lastQuestion": question,
        },
    }


def test_story_page_request_round_trips_strict_camel_case() -> None:
    payload = request_payload()

    request = StoryPageGenerateRequest.model_validate(payload)

    assert request.schema_version == 2
    assert request.page_plan.required_characters == ["tortoise"]
    assert request.generation_profile.skills[1].code == "PHONO_LIAISON"
    assert request.model_dump(by_alias=True) == payload


def test_story_page_response_round_trips_with_quality_and_provenance() -> None:
    payload = response_payload(branching=True)

    response = StoryPageGenerateResponse.model_validate(payload)

    assert response.page.requires_branch_input is True
    assert response.quality.per_skill[0].occurrences == 1
    assert response.generation.repair_accepted is False
    assert response.model_dump(by_alias=True) == payload


@pytest.mark.parametrize(
    "invalid_code",
    [
        "STRUCTURE_CVC",
        "PHONO_REVIEW_REQUIRED",
        "PHONO_UNKNOWN",
        "ONSET_A",
        "CODA_",
    ],
)
def test_request_rejects_skill_codes_not_emitted_by_current_analyzer(
    invalid_code: str,
) -> None:
    payload = request_payload()
    payload["generationProfile"]["skills"][0]["code"] = invalid_code

    with pytest.raises(ValidationError):
        StoryPageGenerateRequest.model_validate(payload)


@pytest.mark.parametrize(
    "supported_code",
    [
        "ONSET_ㄸ",
        "NUCLEUS_ㅢ",
        "CODA_ㄺ",
        "HAS_COMPLEX_CODA",
        "PHONO_GLIDE_REDUCTION",
    ],
)
def test_request_accepts_each_supported_skill_code_family(
    supported_code: str,
) -> None:
    payload = request_payload()
    payload["generationProfile"]["skills"][0]["code"] = supported_code

    request = StoryPageGenerateRequest.model_validate(payload)

    assert request.generation_profile.skills[0].code == supported_code


def test_request_rejects_snake_case_and_unknown_nested_fields() -> None:
    payload = request_payload()
    payload["request_id"] = payload.pop("requestId")

    with pytest.raises(ValidationError):
        StoryPageGenerateRequest.model_validate(payload)

    payload = request_payload()
    payload["storyState"]["childName"] = "저장하거나 모델에 전달하면 안 되는 값"

    with pytest.raises(ValidationError):
        StoryPageGenerateRequest.model_validate(payload)


def test_request_rejects_unknown_required_character() -> None:
    payload = request_payload()
    payload["pagePlan"]["requiredCharacters"] = ["unknown-character"]

    with pytest.raises(ValidationError, match="unknown characterId"):
        StoryPageGenerateRequest.model_validate(payload)


def test_request_rejects_invalid_profile_policy() -> None:
    payload = request_payload()
    payload["generationProfile"]["skills"][0]["maxOccurrences"] = None

    with pytest.raises(ValidationError, match="maxOccurrences is required"):
        StoryPageGenerateRequest.model_validate(payload)

    payload = request_payload()
    payload["generationProfile"]["contentContract"][
        "preferredWrittenSyllables"
    ] = {"min": 45, "max": 80}

    with pytest.raises(ValidationError, match="must be inside"):
        StoryPageGenerateRequest.model_validate(payload)


def test_response_enforces_question_choice_contract() -> None:
    payload = response_payload()
    payload["page"]["requiresBranchInput"] = True

    with pytest.raises(ValidationError, match="exactly two choices"):
        StoryPageGenerateResponse.model_validate(payload)

    payload = response_payload()
    payload["page"]["question"] = "질문만 있으면 안 돼요?"

    with pytest.raises(ValidationError, match="must not contain"):
        StoryPageGenerateResponse.model_validate(payload)


def test_response_rejects_accepted_repair_without_attempt() -> None:
    payload = response_payload()
    payload["generation"]["repairAttempted"] = False
    payload["generation"]["repairAccepted"] = True

    with pytest.raises(ValidationError, match="repairAttempted"):
        StoryPageGenerateResponse.model_validate(payload)


def test_page_four_is_the_only_branching_page_for_non_concluding_request() -> None:
    payload = request_payload()
    payload["pagePlan"]["pageNumber"] = 4
    payload["pagePlan"]["questionFocus"] = "거북이가 마지막 언덕에서 무엇을 할지"
    request = StoryPageGenerateRequest.model_validate(payload)
    response = StoryPageGenerateResponse.model_validate(
        response_payload(branching=True)
    )

    assert response.validate_against_request(request) is response

    payload["conclude"] = True
    with pytest.raises(ValidationError, match="allowed only"):
        StoryPageGenerateRequest.model_validate(payload)


def test_response_cross_validation_rejects_branch_before_page_four() -> None:
    request = StoryPageGenerateRequest.model_validate(request_payload())
    response = StoryPageGenerateResponse.model_validate(
        response_payload(branching=True)
    )

    with pytest.raises(ValueError, match="pageNumber"):
        response.validate_against_request(request)


def test_content_contract_rejects_non_production_sentence_count() -> None:
    payload = request_payload()
    payload["generationProfile"]["contentContract"]["sentenceCount"] = 5

    with pytest.raises(ValidationError):
        StoryPageGenerateRequest.model_validate(payload)
