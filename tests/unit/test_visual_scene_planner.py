from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.personalization.page_splitter import (
    DynamicStoryPage,
    partition_chapter_sentences,
)
from iread_ai.personalization.visual_scene import (
    OpenAIVisualScenePlanner,
    VisualSceneGenerationError,
)
from tests.unit.test_story_chapter_contracts import request_payload


def _pages() -> tuple[DynamicStoryPage, ...]:
    return partition_chapter_sentences(
        (
            "토끼가 나무 아래에서 쉬어요.",
            "거북이는 언덕을 천천히 올라요.",
            "“나는 계속 갈 거야.” 거북이가 말해요.",
            "토끼가 눈을 뜨고 깜짝 놀라요.",
            "거북이는 언덕 끝에 먼저 닿아요.",
            "두 친구는 멀리 있는 종을 바라봐요.",
        ),
        min_pages=2,
        max_pages=2,
        preferred_min_syllables=1,
        preferred_max_syllables=100,
        accepted_min_syllables=1,
        accepted_max_syllables=100,
        direct_dialogue_per_page=0,
    ).pages


def _character(
    character_id: str,
    *,
    present: bool,
) -> dict[str, Any]:
    if not present:
        return {
            "characterId": character_id,
            "present": False,
            "position": None,
            "orientation": None,
            "gazeTarget": None,
            "action": None,
            "emotion": None,
        }
    return {
        "characterId": character_id,
        "present": True,
        "position": "숲길 왼쪽",
        "orientation": "언덕 방향",
        "gazeTarget": "언덕 위 종",
        "action": "언덕을 향해 한 걸음 나아감",
        "emotion": {
            "type": "FOCUSED",
            "intensity": "LOW",
        },
    }


def _document() -> dict[str, Any]:
    return {
        "pages": [
            {
                "pageNumber": page_number,
                "visualScene": {
                    "shot": "WIDE_THREE_QUARTER",
                    "characters": [
                        _character("hare", present=page_number == 2),
                        _character("tortoise", present=True),
                    ],
                    "mustInclude": ["숲길", "언덕"],
                    "mustAvoid": [
                        "글자와 말풍선",
                        "같은 캐릭터 중복",
                    ],
                },
            }
            for page_number in (1, 2)
        ]
    }


@pytest.mark.asyncio
async def test_openai_visual_scene_planner_uses_one_strict_call_without_student_id() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": json.dumps(
                    _document(),
                    ensure_ascii=False,
                ),
                "usage": {"input_tokens": 100, "output_tokens": 200},
            },
        )

    request = StoryChapterGenerateRequest.model_validate(request_payload())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        planner = OpenAIVisualScenePlanner(
            api_key="test-key",
            model="gpt-test",
            client=client,
        )
        result = await planner.generate(
            request=request,
            pages=_pages(),
            question="어느 길로 갈까요?",
            choices=("꽃길로 가요.", "돌길로 가요."),
        )

    assert result.api_call_count == 1
    assert result.model == "gpt-test"
    assert len(result.scenes) == 2
    assert result.scenes[0]["characters"][0]["present"] is False
    input_text = json.dumps(captured["input"], ensure_ascii=False)
    assert "studentId" not in input_text
    assert '"student_id"' not in input_text
    schema = captured["text"]["format"]["schema"]
    assert captured["text"]["format"]["strict"] is True
    assert schema["properties"]["pages"]["minItems"] == 2
    assert schema["properties"]["pages"]["maxItems"] == 2


@pytest.mark.asyncio
async def test_openai_visual_scene_planner_rejects_page_reordering() -> None:
    document = _document()
    document["pages"].reverse()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": json.dumps(
                    document,
                    ensure_ascii=False,
                ),
            },
        )

    request = StoryChapterGenerateRequest.model_validate(request_payload())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        planner = OpenAIVisualScenePlanner(
            api_key="test-key",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(
            VisualSceneGenerationError,
            match="invalid visual-scene",
        ):
            await planner.generate(
                request=request,
                pages=_pages(),
                question="어느 길로 갈까요?",
                choices=("꽃길로 가요.", "돌길로 가요."),
            )


@pytest.mark.asyncio
async def test_openai_visual_scene_planner_rejects_emotion_for_absent_character() -> None:
    document = _document()
    document["pages"][0]["visualScene"]["characters"][0]["emotion"] = {
        "type": "CALM",
        "intensity": "LOW",
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": json.dumps(
                    document,
                    ensure_ascii=False,
                ),
            },
        )

    request = StoryChapterGenerateRequest.model_validate(request_payload())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        planner = OpenAIVisualScenePlanner(
            api_key="test-key",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(VisualSceneGenerationError):
            await planner.generate(
                request=request,
                pages=_pages(),
                question="어느 길로 갈까요?",
                choices=("꽃길로 가요.", "돌길로 가요."),
            )


@pytest.mark.asyncio
async def test_openai_visual_scene_planner_marks_rate_limit_retryable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow"}})

    request = StoryChapterGenerateRequest.model_validate(request_payload())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        planner = OpenAIVisualScenePlanner(
            api_key="test-key",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(VisualSceneGenerationError) as captured:
            await planner.generate(
                request=request,
                pages=_pages(),
                question="어느 길로 갈까요?",
                choices=("꽃길로 가요.", "돌길로 가요."),
            )

    assert captured.value.retryable is True
