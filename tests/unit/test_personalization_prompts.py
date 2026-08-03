from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from iread_ai.personalization.generator import (
    PageCandidate,
    PageGenerationContext,
)
from iread_ai.personalization.prompts import (
    build_reading_policy_hints,
    build_repair_user_prompt,
    load_repair_prompt,
)


@dataclass(frozen=True)
class Profile:
    def to_dict(self) -> dict[str, Any]:
        return {
            "contentContract": {"sentenceCount": 4},
            "skills": [
                {
                    "code": "ONSET_ㄲ",
                    "role": "LIMITED",
                    "maxOccurrences": 1,
                },
                {
                    "code": "PHONO_LIAISON",
                    "role": "TARGET",
                    "targetMin": 1,
                    "targetMax": 1,
                },
            ],
        }


def test_repair_prompt_preserves_child_detour_and_return_event() -> None:
    prompt = load_repair_prompt()

    assert "1~3문장은 아이의 답이 만든 하나의" in prompt
    assert "`locked_event`의 본래 흐름으로 돌아가야 합니다" in prompt


def test_reading_policy_hints_describe_profile_roles() -> None:
    hints = build_reading_policy_hints(Profile())

    assert hints["limited"] == [
        {
            "code": "ONSET_ㄲ",
            "description": "초성이 ㄲ인 한글 음절",
            "min_occurrences": 0,
            "max_occurrences": 1,
        }
    ]
    assert hints["target"][0]["code"] == "PHONO_LIAISON"
    assert hints["target"][0]["min_occurrences"] == 1


def test_repair_user_prompt_contains_locks_source_and_plan() -> None:
    context = PageGenerationContext(
        story_title="별빛 숲",
        story_context="토끼가 별빛을 찾고 있어요.",
        locked_event="토끼가 작은 별을 만나요.",
        child_input="별에게 노래를 불러 줘.",
    )
    source = PageCandidate(
        candidate_id="candidate-1",
        sentences=(
            "토끼가 숲으로 가요.",
            "작은 별이 빛나요.",
            "두 친구가 웃어요.",
            "길이 다시 보여요.",
        ),
    )

    document = json.loads(
        build_repair_user_prompt(
            context=context,
            profile=Profile(),
            source_candidate=source,
            repair_plan={
                "editable_sentence_indexes": [2],
                "violations": [],
            },
        )
    )

    assert document["task"] == "repair_story_page"
    assert document["source_candidate"]["candidate_id"] == "candidate-1"
    assert document["page_locks"]["locked_event"] == (
        "토끼가 작은 별을 만나요."
    )
    assert document["repair_plan"]["editable_sentence_indexes"] == [2]
