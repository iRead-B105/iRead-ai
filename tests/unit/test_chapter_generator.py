from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from iread_ai.personalization.chapter_generator import (
    ChapterCandidate,
    ChapterGenerationContext,
    ChapterGenerationError,
    MockChapterCandidateGenerator,
    OpenAIChapterCandidateGenerator,
    build_chapter_user_prompt,
    load_chapter_prompt,
)
from iread_ai.personalization.prompts import BASELINE_PROMPT_MODE

_BASE_URL = "https://openai.invalid/v1"
_PROFILE = {
    "content_contract": {
        "preferred_written_syllables": {"min": 50, "max": 70},
        "accepted_written_syllables": {"min": 45, "max": 80},
    },
    "skills": [
        {
            "code": "PHONO_LIAISON",
            "role": "EXCLUDED",
            "max_occurrences": 0,
        }
    ],
    "protected_terms": ["토끼", "거북이"],
}


def _context(
    *,
    child_input: str = "방구 소리",
    conclude: bool = False,
) -> ChapterGenerationContext:
    return ChapterGenerationContext(
        story_title="토끼와 거북이",
        story_context="토끼와 거북이가 숲속에서 경주해요.",
        chapter_goal="경주가 시작되고 두 친구의 속도 차이가 드러나요.",
        ordered_events=(
            "출발 신호가 울리고 토끼가 먼저 달려요.",
            "거북이는 자기 걸음으로 계속 길을 가요.",
        ),
        chapter_number=2,
        child_input=child_input,
        last_question="거북이가 어떤 리듬 말을 외칠까요?",
        previous_context=("두 친구가 경주를 하기로 약속했어요.",),
        characters=("토끼", "거북이"),
        question_focus=None if conclude else "거북이가 고를 작은 도움",
        conclude=conclude,
    )


def _candidate(
    candidate_number: int,
    *,
    child_branch_active: bool,
    branch_required: bool,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "candidate_id": f"candidate-{candidate_number}",
        "sentences": [
            "방구 소리가 출발선에서 크게 울려요.",
            "토끼는 깜짝 놀라 앞으로 폴짝 뛰어요.",
            "거북이는 웃다가 등에 붙은 잎을 떼어요.",
            "두 친구는 다시 경주 길로 힘차게 나가요.",
            "토끼는 벌써 낮은 언덕 위까지 달려가요.",
            "거북이는 자기 걸음으로 천천히 따라가요.",
            "토끼가 “여기까지 와 봐!” 하고 크게 외쳐요.",
            "거북이는 흔들림 없이 돌길을 건너가요.",
        ],
        "child_detour_end_sentence_index": (
            3 if child_branch_active else None
        ),
    }
    if branch_required:
        document.update(
            {
                "question": "거북이는 돌길에서 무엇을 쓸까요?",
                "subtitle": "돌길에서 고를 작은 도움",
                "choices": [
                    "긴 막대를 써요.",
                    "넓은 잎을 밟아요.",
                    "짧은 밧줄을 잡아요.",
                ],
            }
        )
    return document


def _completed_response(
    candidate_count: int,
    *,
    child_branch_active: bool = True,
    branch_required: bool = True,
) -> dict[str, Any]:
    document = {
        "candidates": [
            _candidate(
                candidate_number,
                child_branch_active=child_branch_active,
                branch_required=branch_required,
            )
            for candidate_number in range(1, candidate_count + 1)
        ]
    }
    return {
        "status": "completed",
        "output_text": json.dumps(document, ensure_ascii=False),
        "usage": {
            "input_tokens": 180,
            "output_tokens": 500,
            "total_tokens": 680,
        },
    }


def _generator(
    client: httpx.AsyncClient,
) -> OpenAIChapterCandidateGenerator:
    return OpenAIChapterCandidateGenerator(
        api_key="test-key",
        model="test-model",
        base_url=_BASE_URL,
        timeout_seconds=1.0,
        max_output_tokens=1200,
        client=client,
    )


def test_context_and_candidate_validate_dynamic_chapter_contract() -> None:
    context = _context()
    candidate = ChapterCandidate(
        candidate_id="candidate-1",
        sentences=tuple(
            str(sentence)
            for sentence in _candidate(
                1,
                child_branch_active=True,
                branch_required=True,
            )["sentences"]
        ),
        child_detour_end_sentence_index=3,
        question="거북이는 무엇을 쓸까요?",
        choices=("막대를 써요.", "잎을 밟아요.", "밧줄을 잡아요."),
    )

    branch_plan = context.to_dict()["child_branch_plan"]
    assert branch_plan["active"] is True
    assert branch_plan["answer_owner_candidates"] == ["거북이"]
    assert branch_plan["answer_delivery"] == "natural_direct_dialogue"
    assert candidate.child_detour_end_sentence_index == 3
    assert len(candidate.sentences) == 8

    with pytest.raises(ValueError, match="between 8 and 16"):
        ChapterCandidate(
            candidate_id="too-short",
            sentences=("가요.",) * 5,
        )
    with pytest.raises(ValueError, match="must be 3 or 4"):
        ChapterCandidate(
            candidate_id="wrong-detour",
            sentences=("가요.",) * 8,
            child_detour_end_sentence_index=2,
        )


def test_prompt_is_utf8_and_user_document_contains_context_and_profile() -> None:
    system_prompt = load_chapter_prompt()
    user_document = json.loads(
        build_chapter_user_prompt(
            context=_context(),
            profile=_PROFILE,
        )
    )

    assert "한 장 전체" in system_prompt
    assert "페이지 번호나 페이지 묶음을" in system_prompt
    assert "난 금방 이길 거야!" in system_prompt
    assert "대사를 0개 또는 1개" in system_prompt
    assert "무조건 2문장에 배치하지" in system_prompt
    assert "본문 마지막 문장 다음" in system_prompt
    assert "이미 울린 신호" in system_prompt
    assert "나무 구호" in system_prompt
    assert "크게 외치기" in system_prompt
    assert user_document["chapter"]["child_input"] == "방구 소리"
    assert user_document["chapter"]["ordered_events"]
    assert user_document["generation_profile"]["skills"]
    assert user_document["reading_policy_hints"]["excluded"][0][
        "description"
    ]


def test_baseline_prompt_omits_the_reading_profile() -> None:
    system_prompt = load_chapter_prompt(
        prompt_mode=BASELINE_PROMPT_MODE,
    )
    user_document = json.loads(
        build_chapter_user_prompt(
            context=_context(),
            profile=None,
            prompt_mode=BASELINE_PROMPT_MODE,
        )
    )

    assert "일반 한국어 이야기 한 장 생성" in system_prompt
    assert user_document["chapter"]["child_input"] == "방구 소리"
    assert "generation_profile" not in user_document
    assert "reading_policy_hints" not in user_document


@pytest.mark.asyncio
async def test_baseline_generator_makes_one_call_without_profile_hints() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completed_response(1))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        batch = await _generator(client).generate(
            _context(),
            None,
            candidate_count=1,
            prompt_mode=BASELINE_PROMPT_MODE,
        )

    assert len(batch.candidates) == 1
    user_document = json.loads(
        captured["input"][1]["content"][0]["text"]
    )
    assert "generation_profile" not in user_document
    assert "reading_policy_hints" not in user_document
    assert "일반 한국어 이야기" in (
        captured["input"][0]["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_openai_generator_requests_three_whole_chapter_candidates_once() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_completed_response(3))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        batch = await _generator(client).generate(
            _context(),
            _PROFILE,
            candidate_count=3,
        )

    assert len(requests) == 1
    schema = requests[0]["text"]["format"]["schema"]
    candidates_schema = schema["properties"]["candidates"]
    item_schema = candidates_schema["items"]
    assert candidates_schema["minItems"] == 3
    assert candidates_schema["maxItems"] == 3
    assert item_schema["properties"]["sentences"]["minItems"] == 9
    assert item_schema["properties"]["sentences"]["maxItems"] == 9
    assert item_schema["properties"]["sentences"]["items"]["pattern"] == (
        "[가-힣]"
    )
    assert item_schema["properties"][
        "child_detour_end_sentence_index"
    ]["enum"] == [3]
    assert {"question", "choices"} <= set(item_schema["required"])
    assert item_schema["properties"]["choices"]["minItems"] == 3
    assert item_schema["properties"]["choices"]["maxItems"] == 3
    assert len(batch.candidates) == 3
    assert batch.candidates[0].child_detour_end_sentence_index == 3
    assert batch.candidates[0].question
    assert batch.usage["total_tokens"] == 680
    assert requests[0]["max_output_tokens"] == 1200


@pytest.mark.asyncio
async def test_concluding_chapter_has_no_branch_fields_or_detour() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_completed_response(
                1,
                child_branch_active=False,
                branch_required=False,
            ),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        batch = await _generator(client).generate(
            _context(child_input="", conclude=True),
            _PROFILE,
            candidate_count=1,
        )

    item_schema = captured["text"]["format"]["schema"]["properties"][
        "candidates"
    ]["items"]
    assert item_schema["properties"][
        "child_detour_end_sentence_index"
    ] == {"type": "null"}
    assert "question" not in item_schema["properties"]
    assert "choices" not in item_schema["properties"]
    assert batch.candidates[0].child_detour_end_sentence_index is None
    assert batch.candidates[0].question is None
    assert batch.candidates[0].choices == ()


@pytest.mark.asyncio
async def test_mock_generator_obeys_count_and_child_detour_contract() -> None:
    batch = await MockChapterCandidateGenerator().generate(
        _context(),
        _PROFILE,
        candidate_count=3,
    )

    assert len(batch.candidates) == 3
    assert len({candidate.candidate_id for candidate in batch.candidates}) == 3
    assert all(
        8 <= len(candidate.sentences) <= 16
        for candidate in batch.candidates
    )
    assert all(
        candidate.child_detour_end_sentence_index == 3
        for candidate in batch.candidates
    )
    assert all(candidate.question for candidate in batch.candidates)


@pytest.mark.asyncio
async def test_mock_generator_uses_each_storys_planned_events() -> None:
    generator = MockChapterCandidateGenerator()
    race_batch = await generator.generate(
        _context(child_input=""),
        _PROFILE,
        candidate_count=1,
    )
    ant_context = ChapterGenerationContext(
        story_title="개미와 베짱이",
        story_context="여름 들판에서 두 친구가 서로 다른 하루를 보내요.",
        chapter_goal="개미와 베짱이가 처음 만나 서로의 일을 알게 돼요.",
        ordered_events=(
            "개미가 더운 들판에서 씨앗을 옮겨요.",
            "베짱이가 풀잎 위에서 노래하다 개미를 만나요.",
        ),
        characters=("개미", "베짱이"),
        question_focus="둘이 함께 해 볼 재미있는 일",
    )
    ant_batch = await generator.generate(
        ant_context,
        _PROFILE,
        candidate_count=1,
    )

    race_text = " ".join(race_batch.candidates[0].sentences)
    ant_text = " ".join(ant_batch.candidates[0].sentences)
    assert race_text != ant_text
    assert "출발 신호" in race_text
    assert "씨앗" in ant_text


@pytest.mark.asyncio
async def test_generator_rejects_missing_child_detour_boundary() -> None:
    response = _completed_response(1)
    raw_document = json.loads(response["output_text"])
    raw_document["candidates"][0][
        "child_detour_end_sentence_index"
    ] = None
    response["output_text"] = json.dumps(
        raw_document,
        ensure_ascii=False,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(
            ChapterGenerationError,
            match="invalid story-chapter",
        ):
            await _generator(client).generate(
                _context(),
                _PROFILE,
                candidate_count=1,
            )
