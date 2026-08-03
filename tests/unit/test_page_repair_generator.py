from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from iread_ai.personalization.generator import (
    MockPageCandidateGenerator,
    OpenAIPageCandidateGenerator,
    PageCandidate,
    PageGenerationContext,
)


@dataclass(frozen=True)
class DummyProfile:
    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": [
                {
                    "code": "CODA_ㅆ",
                    "role": "EXCLUDED",
                    "max_occurrences": 0,
                }
            ],
            "protected_terms": ["토끼"],
        }


def _context() -> PageGenerationContext:
    return PageGenerationContext(
        story_title="토끼와 거북이",
        story_context="두 친구가 숲길에서 경주해요.",
        locked_event="거북이가 응원을 듣고 다시 걸어요.",
        page_number=2,
        child_input="천천히 가도 괜찮아!",
        previous_pages=("두 친구가 경주를 시작했어요.",),
        characters=("토끼", "거북이"),
    )


def _source() -> PageCandidate:
    return PageCandidate(
        candidate_id="candidate-1",
        sentences=(
            "거북이는 응원을 듣고 다시 걸어요.",
            "토끼는 거북이를 향해 힘차게 응원해요.",
            "거북이는 따뜻한 말에 힘을 내요.",
            "두 친구 앞에 넓은 길이 이어져요.",
        ),
    )


def _repair_plan() -> dict[str, Any]:
    return {
        "editable_sentence_indexes": [2, 3],
        "max_changed_sentences": 2,
        "trigger_reasons": ["DIRECT_DIALOGUE_COUNT"],
        "violations": [],
    }


@pytest.mark.asyncio
async def test_openai_repair_uses_locked_schema_and_parses_result() -> None:
    captured: dict[str, Any] = {}
    document = {
        "source_candidate_id": "candidate-1",
        "repair_status": "REPAIRED",
        "replacements": [
            {
                "sentence_index": 2,
                "sentence": "토끼는 “우리 천천히 가자!”라고 말해요.",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": json.dumps(
                    document,
                    ensure_ascii=False,
                ),
                "usage": {"total_tokens": 90},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        generator = OpenAIPageCandidateGenerator(
            api_key="test-key",
            model="test-model",
            base_url="https://openai.invalid/v1",
            timeout_seconds=1.0,
            max_output_tokens=900,
            client=client,
        )
        batch = await generator.repair(
            _context(),
            DummyProfile(),
            _source(),
            repair_plan=_repair_plan(),
        )

    schema = captured["text"]["format"]["schema"]
    indexes = schema["properties"]["replacements"]["items"][
        "properties"
    ]["sentence_index"]["enum"]
    assert indexes == [2, 3]
    assert captured["text"]["format"]["strict"] is True
    assert batch.repair_status == "REPAIRED"
    assert batch.replacements[0].sentence_index == 2
    assert "repair_story_page" in batch.user_prompt
    assert batch.usage["total_tokens"] == 90


@pytest.mark.asyncio
async def test_mock_repair_uses_confirmed_child_input() -> None:
    batch = await MockPageCandidateGenerator().repair(
        _context(),
        DummyProfile(),
        _source(),
        repair_plan=_repair_plan(),
    )

    assert batch.repair_status == "REPAIRED"
    assert "천천히 가도 괜찮아!" in batch.replacements[0].sentence


@pytest.mark.asyncio
async def test_openai_repair_rejects_plan_without_editable_sentence() -> None:
    generator = OpenAIPageCandidateGenerator(
        api_key="test-key",
        model="test-model",
    )

    with pytest.raises(ValueError, match="editable sentence"):
        await generator.repair(
            _context(),
            DummyProfile(),
            _source(),
            repair_plan={"editable_sentence_indexes": []},
        )
