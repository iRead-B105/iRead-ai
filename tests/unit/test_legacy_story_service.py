from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from iread_ai.application.legacy_story_service import LegacyStoryGenerationService
from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.generation_models import ContinueStoryRequest, GenerateStoryRequest


@dataclass
class _Page:
    sentences: list[str]
    question: str | None = None
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


@pytest.mark.asyncio
async def test_opening_maps_personalized_chapter_to_legacy_five_lines() -> None:
    sentences = [f"첫 장 문장 {index}." for index in range(1, 9)]
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:4]),
            _Page(
                sentences=sentences[4:],
                question="거북이는 이제 무엇을 할까요?",
                choices=["계속 걸어요.", "노래해요.", "토끼를 불러요."],
            ),
        ]
    )
    generator = _RecordingChapterService(chapter)
    service = LegacyStoryGenerationService(chapter_service=generator)

    response = await service.generate(_opening_request())

    assert len(response.lines) == 5
    assert [line.requiresBranchInput for line in response.lines] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert " ".join(line.content for line in response.lines[:4]) == " ".join(sentences)
    assert response.lines[-1].content == "거북이는 이제 무엇을 할까요?"
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
    sentences = [f"마지막 장 문장 {index}." for index in range(1, 11)]
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:4]),
            _Page(sentences=sentences[4:8]),
            _Page(sentences=sentences[8:]),
        ]
    )
    generator = _RecordingChapterService(chapter)
    service = LegacyStoryGenerationService(chapter_service=generator)
    payload = {
        **_opening_request().model_dump(mode="json"),
        "requestId": "legacy-continue",
        "currentProgress": 50,
        "currentStoryLineId": 5,
        "branchIntent": "큰 소리로 응원해요.",
        "history": [
            {
                "storyLineId": index,
                "content": f"앞 이야기 {index}.",
                "requiresBranchInput": index == 5,
            }
            for index in range(1, 6)
        ],
    }
    request = ContinueStoryRequest.model_validate(payload)

    response = await service.continue_story(request)

    assert response.completed is True
    assert response.nextProgress == 100
    assert len(response.lines) == 5
    assert " ".join(line.content for line in response.lines) == " ".join(sentences)
    assert all(line.branchPrompt is None for line in response.lines)

    chapter_request = generator.requests[0]
    assert chapter_request.conclude is True
    assert chapter_request.story_revision == 1
    assert chapter_request.branch_input is not None
    assert chapter_request.branch_input.source == "TEXT_CONFIRMED"
    assert chapter_request.branch_input.text == "큰 소리로 응원해요."
    assert chapter_request.story_state.last_question == "앞 이야기 5."
    assert "앞 이야기 1." in chapter_request.story_state.rolling_summary
    assert len(chapter_request.story_state.recent_pages) == 4


@pytest.mark.asyncio
async def test_choice_labels_are_kept_unique_inside_legacy_limit() -> None:
    sentences = [f"문장 {index}." for index in range(1, 9)]
    duplicate = "아주 긴 선택지" * 20
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:4]),
            _Page(
                sentences=sentences[4:],
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


@pytest.mark.asyncio
async def test_success_logs_quality_without_story_or_branch_text() -> None:
    sentences = [f"로그 문장 {index}." for index in range(1, 9)]
    chapter = _Chapter(
        [
            _Page(sentences=sentences[:4]),
            _Page(
                sentences=sentences[4:],
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
