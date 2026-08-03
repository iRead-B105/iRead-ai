from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from iread_ai.application.legacy_story_service import (
    LegacyStoryGenerationService,
    _validate_branch_question,
    _validate_page_sentences,
)
from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.generation_models import ContinueStoryRequest, GenerateStoryRequest


@dataclass
class _Page:
    sentences: list[str]
    question: str | None = None
    subtitle: str | None = None
    choices: list[str] | None = None


class _Chapter:
    def __init__(self, pages: list[_Page]) -> None:
        self.pages = pages

    def validate_against_request(
        self,
        request: StoryChapterGenerateRequest,
    ) -> _Chapter:
        self.request = request
        return self

    def model_dump(self, *, by_alias: bool, mode: str) -> dict[str, object]:
        assert by_alias is True
        assert mode == "json"
        return {
            "generationId": "generation-test",
            "chapterNumber": 1,
            "quality": {
                "chapter": {
                    "status": "PASS",
                    "analysisStatus": "FULL",
                    "contractPass": True,
                    "contractFailures": [],
                    "writtenSyllableCount": 61,
                    "directDialogueCount": 1,
                    "excludedOverageCount": 0,
                    "limitedOverageCount": 0,
                    "riskPer10": 0.8,
                    "perSkill": [
                        {
                            "code": "PHONO_LIAISON",
                            "role": "LIMITED",
                            "status": "PASS",
                            "occurrences": 2,
                            "maxOccurrences": 4,
                            "targetMin": None,
                            "targetMax": None,
                            "overage": 0,
                            "targetDistance": None,
                            "weightedRisk": 0.0,
                        }
                    ],
                },
                "pages": [],
            },
            "generation": {
                "provider": "mock",
                "model": "mock-story-v1",
                "promptVersion": "test-prompt",
                "candidateCount": 3,
                "selectedCandidateId": "2",
                "apiCallCount": 2,
                "repairAttempted": True,
                "repairAccepted": True,
                "changedSentences": [],
                "repairDecisionReasons": ["RISK_REDUCED"],
            },
            "timingMs": {
                "generation": 1200.0,
                "analysis": 80.0,
                "pagination": 1.0,
                "repair": 400.0,
                "visualScene": 0.0,
                "total": 1681.0,
            },
        }


class _RecordingChapterService:
    def __init__(self, chapter: _Chapter) -> None:
        self.chapter = chapter
        self.requests: list[StoryChapterGenerateRequest] = []

    async def generate(self, request: StoryChapterGenerateRequest) -> _Chapter:
        self.requests.append(request)
        return self.chapter


def _opening_request() -> GenerateStoryRequest:
    return GenerateStoryRequest.model_validate(
        {
            "requestId": "legacy-opening",
            "storyId": 101,
            "studentId": 7,
            "schemaVersion": 1,
            "currentProgress": 0,
            "storyTemplate": {
                "storyTemplateId": 11,
                "title": "토끼와 거북이",
                "context": "토끼가 거북이를 놀리고 숲길 경주를 시작한다.",
            },
        }
    )


def _story_sentence(index: int) -> str:
    variants = (
        "토끼와 거북이는 숲길을 함께 걸어갔어요",
        "햇살 아래 두 친구는 숲길을 천천히 살폈지요",
        "곧 나뭇잎 사이로 작은 길이 모습을 드러냈네요",
    )
    return f"{variants[(index - 1) % len(variants)]} {index}."


@pytest.mark.asyncio
async def test_opening_maps_personalized_chapter_to_first_four_pages() -> None:
    sentences = [_story_sentence(index) for index in range(1, 10)]
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:3]),
            _Page(sentences=sentences[3:6]),
            _Page(
                sentences=sentences[6:],
                question="거북이는 이제 무엇을 할까요?",
                subtitle="경주가 시작되는 숲길",
                choices=["계속 걸어요.", "노래해요.", "토끼를 불러요."],
            ),
        ]
    )
    generator = _RecordingChapterService(chapter)
    service = LegacyStoryGenerationService(chapter_service=generator)

    response = await service.generate(_opening_request())

    assert len(response.lines) == 4
    assert [line.requiresBranchInput for line in response.lines] == [
        False,
        False,
        False,
        True,
    ]
    assert " ".join(line.content for line in response.lines[:3]) == " ".join(sentences)
    assert response.nextProgress == 4
    assert response.lines[-1].content == "거북이는 이제 무엇을 할까요?"
    assert response.lines[-1].branchPrompt.subtitle == "경주가 시작되는 숲길"
    assert [option.label for option in response.lines[-1].branchPrompt.options] == [
        "계속 걸어요.",
        "노래해요.",
        "토끼를 불러요.",
    ]

    request = generator.requests[0]
    assert request.conclude is False
    assert request.story_revision == 0
    assert [character.name for character in request.story_state.characters] == [
        "토끼",
        "거북이",
    ]
    assert request.generation_profile.skills
    assert request.student_id == 7


@pytest.mark.asyncio
async def test_continue_preserves_full_history_and_confirmed_branch_input() -> None:
    sentences = [_story_sentence(index) for index in range(1, 13)]
    sentences[0] = "거북이는 큰 응원을 듣고 힘차게 발을 내디뎠어요."
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:3]),
            _Page(sentences=sentences[3:6]),
            _Page(sentences=sentences[6:9]),
            _Page(
                sentences=sentences[9:],
                question="이제 어떻게 할까요?",
                choices=["응원해요.", "함께 가요.", "잠깐 쉬어요."],
            ),
        ]
    )
    generator = _RecordingChapterService(chapter)
    service = LegacyStoryGenerationService(chapter_service=generator)
    payload = {
        **_opening_request().model_dump(mode="json"),
        "requestId": "legacy-continue",
        "currentProgress": 4,
        "currentStoryLineId": 4,
        "branchIntent": "큰 소리로 응원해요.",
        "history": [
            {
                "storyLineId": index,
                "content": f"앞 이야기 {index}.",
                "requiresBranchInput": index == 4,
            }
            for index in range(1, 5)
        ],
    }
    request = ContinueStoryRequest.model_validate(payload)

    response = await service.continue_story(request)

    assert response.completed is False
    assert response.nextProgress == 9
    assert len(response.lines) == 5
    assert response.lines[0].content.startswith(
        "거북이는 큰 응원을 듣고 힘차게 발을 내디뎠어요."
    )
    assert "선택으로 새 길이" not in response.lines[0].content
    assert " ".join(line.content for line in response.lines[:4]).endswith(sentences[-1])
    assert response.lines[-1].branchPrompt is not None

    chapter_request = generator.requests[0]
    assert chapter_request.conclude is False
    assert chapter_request.story_revision == 4
    assert chapter_request.branch_input is not None
    assert chapter_request.branch_input.source == "TEXT_CONFIRMED"
    assert chapter_request.branch_input.text == "큰 소리로 응원해요."
    assert chapter_request.story_state.last_question == "앞 이야기 4."
    assert "앞 이야기 1." in chapter_request.story_state.rolling_summary
    assert len(chapter_request.story_state.recent_pages) == 4


@pytest.mark.asyncio
async def test_continue_integrates_confirmed_branch_as_a_natural_event() -> None:
    sentences = [_story_sentence(index) for index in range(1, 13)]
    sentences[0] = "거북이는 다친 새와 별빛 다리를 함께 건넜어요."
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:3]),
            _Page(sentences=sentences[3:6]),
            _Page(sentences=sentences[6:9]),
            _Page(
                sentences=sentences[9:],
                question="이제 어떻게 할까요?",
                choices=["응원해요.", "함께 가요.", "잠깐 쉬어요."],
            ),
        ]
    )
    service = LegacyStoryGenerationService(
        chapter_service=_RecordingChapterService(chapter)
    )
    intent = "다친 새와 함께 별빛 다리를 건널래"
    request = ContinueStoryRequest.model_validate(
        {
            **_opening_request().model_dump(mode="json"),
            "requestId": "legacy-verbatim-branch",
            "currentProgress": 4,
            "currentStoryLineId": 4,
            "branchIntent": intent,
            "history": [
                {
                    "storyLineId": index,
                    "content": f"앞 이야기 {index}.",
                    "requiresBranchInput": index == 4,
                }
                for index in range(1, 5)
            ],
        }
    )

    response = await service.continue_story(request)

    assert response.lines[0].content.startswith(
        "거북이는 다친 새와 별빛 다리를 함께 건넜어요."
    )
    assert not response.lines[0].content.startswith(f"{intent}.")


@pytest.mark.asyncio
async def test_choice_labels_are_kept_unique_inside_legacy_limit() -> None:
    sentences = [_story_sentence(index) for index in range(1, 10)]
    duplicate = "아주 긴 선택지" * 20
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:3]),
            _Page(sentences=sentences[3:6]),
            _Page(
                sentences=sentences[6:],
                question="무엇을 할까요?",
                choices=[duplicate, duplicate, duplicate],
            ),
        ]
    )
    service = LegacyStoryGenerationService(chapter_service=_RecordingChapterService(chapter))

    response = await service.generate(_opening_request())
    labels = [option.label for option in response.lines[-1].branchPrompt.options]

    assert len(set(labels)) == 3
    assert all(len(label) <= 80 for label in labels)


def test_story_body_rejects_reader_facing_question() -> None:
    with pytest.raises(ValueError, match="reader-facing branch question"):
        _validate_page_sentences(
            (
                "거북이는 노란 나뭇잎을 펼쳤어요.",
                "바람이 잎 가장자리를 세게 밀었지요.",
                "거북이는 이제 어느 길로 갈까요?",
            )
        )


def test_branch_question_must_advance_from_previous_question() -> None:
    with pytest.raises(ValueError, match="advance beyond"):
        _validate_branch_question(
            "거북이는 무엇을 사용할까요?",
            previous_question="거북이는 무엇을 사용할까요?",
        )


@pytest.mark.asyncio
async def test_success_logs_quality_without_story_or_branch_text() -> None:
    sentences = [_story_sentence(index) for index in range(1, 10)]
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:3]),
            _Page(sentences=sentences[3:6]),
            _Page(
                sentences=sentences[6:],
                question="어디로 갈까요?",
                choices=["왼쪽으로 가요.", "오른쪽으로 가요.", "기다려요."],
            ),
        ]
    )
    service = LegacyStoryGenerationService(
        chapter_service=_RecordingChapterService(chapter)
    )

    with patch(
        "iread_ai.application.legacy_story_service.quality_logger.info"
    ) as log_info:
        await service.generate(_opening_request())

    event = json.loads(log_info.call_args.args[1])
    assert event["event"] == "story_generation_quality"
    assert event["operation"] == "GENERATE"
    assert event["quality"]["chapter"]["riskPer10"] == 0.8
    assert event["generation"]["candidateCount"] == 3
    assert event["generation"]["repairAccepted"] is True
    serialized = json.dumps(event, ensure_ascii=False)
    assert "토끼가 거북이를 놀리고" not in serialized
    assert "어디로 갈까요" not in serialized
