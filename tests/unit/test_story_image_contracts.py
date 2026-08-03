from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from iread_ai.contracts.story_image import StoryImageGenerateRequest


def image_request_payload() -> dict[str, Any]:
    return {
        "requestId": "image-page-101-2-1",
        "schemaVersion": 1,
        "storyId": 101,
        "storyRevision": 8,
        "chapterNumber": 2,
        "pageNumber": 1,
        "sentences": [
            "토끼가 출발선에서 웃어요.",
            "거북이는 종을 바라봐요.",
            "두 친구가 출발을 기다려요.",
        ],
        "visualScene": {
            "shot": "WIDE_THREE_QUARTER",
            "characters": [
                {
                    "characterId": "hare",
                    "present": True,
                    "position": "출발선 왼쪽",
                    "orientation": "결승선 방향",
                    "gazeTarget": "finishBell",
                    "action": "두 발을 모으고 출발을 기다림",
                    "emotion": {
                        "type": "EXCITED",
                        "intensity": "MEDIUM",
                    },
                }
            ],
            "mustInclude": ["출발선", "언덕 너머의 종"],
            "mustAvoid": ["글자", "테두리"],
        },
        "storyContext": {
            "title": "토끼와 거북이",
            "characters": [
                {
                    "characterId": "hare",
                    "name": "토끼",
                    "role": "주인공",
                    "immutableTraits": ["하얀 털", "긴 귀"],
                },
                {
                    "characterId": "tortoise",
                    "name": "거북이",
                    "role": "주인공",
                    "immutableTraits": ["초록 등딱지", "차분한 성격"],
                },
            ],
        },
        "characterReferences": [{"characterId": "hare"}],
    }


def test_visual_scene_may_contain_only_current_page_characters() -> None:
    request = StoryImageGenerateRequest.model_validate(image_request_payload())

    assert len(request.story_context.characters) == 2
    assert [character.character_id for character in request.visual_scene.characters] == [
        "hare"
    ]


def test_character_references_are_optional_for_text_only_characters() -> None:
    payload = image_request_payload()
    payload.pop("characterReferences")

    request = StoryImageGenerateRequest.model_validate(payload)

    assert request.character_references == []


def test_unknown_visual_scene_character_is_rejected() -> None:
    payload = image_request_payload()
    payload["visualScene"]["characters"][0]["characterId"] = "dragon"

    with pytest.raises(ValidationError, match="unknown characterId"):
        StoryImageGenerateRequest.model_validate(payload)


def test_duplicate_character_references_are_rejected() -> None:
    payload = image_request_payload()
    payload["characterReferences"] = [
        {"characterId": "hare"},
        {"characterId": "hare"},
    ]

    with pytest.raises(ValidationError, match="must be unique"):
        StoryImageGenerateRequest.model_validate(payload)


def test_client_image_data_and_file_paths_are_not_part_of_v1_contract() -> None:
    for field, value in (
        ("previousImageBase64", "iVBORw0KGgo="),
        ("referenceImagePath", "C:/private/child.png"),
    ):
        payload = deepcopy(image_request_payload())
        payload[field] = value
        with pytest.raises(ValidationError, match="Extra inputs"):
            StoryImageGenerateRequest.model_validate(payload)
