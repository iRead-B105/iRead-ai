from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from typing import Protocol

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.generation_models import (
    ContinueStoryRequest,
    GeneratedStoryLine,
    GenerateStoryRequest,
    GenerateStoryResponse,
    StoryBranchOption,
    StoryBranchPrompt,
)

quality_logger = logging.getLogger("uvicorn.error")


class StoryChapterGenerator(Protocol):
    async def generate(
        self,
        request: StoryChapterGenerateRequest,
    ) -> StoryChapterGenerateResponse: ...


_KNOWN_CHARACTER_NAMES = (
    "토끼",
    "거북이",
    "개미",
    "베짱이",
    "사자",
    "생쥐",
    "여우",
    "두루미",
    "바람",
    "펭귄",
    "공룡",
    "다람쥐",
    "늑대",
    "염소",
    "돼지",
    "병아리",
    "곰",
    "강아지",
    "고양이",
    "아이",
    "할머니",
    "할아버지",
    "공주",
    "왕자",
    "나무꾼",
    "여행자",
    "친구",
)


class LegacyStoryGenerationService:
    def __init__(self, *, chapter_service: StoryChapterGenerator) -> None:
        self._chapter_service = chapter_service

    async def generate(
        self,
        request: GenerateStoryRequest,
    ) -> GenerateStoryResponse:
        chapter_request = _build_chapter_request(request, conclude=False)
        chapter = await self._chapter_service.generate(chapter_request)
        chapter = chapter.validate_against_request(chapter_request)
        narrative_lines = _balanced_lines(
            _chapter_sentences(chapter),
            line_count=4,
        )
        final_page = chapter.pages[-1]
        if final_page.question is None or len(final_page.choices) != 3:
            raise ValueError("opening chapter must end with one question and three choices")
        branch_prompt = StoryBranchPrompt(
            options=[
                StoryBranchOption(
                    optionNo=index,
                    label=label,
                )
                for index, label in enumerate(
                    _unique_choice_labels(final_page.choices),
                    start=1,
                )
            ]
        )
        response = GenerateStoryResponse(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            nextProgress=50,
            completed=False,
            lines=[
                *(
                    GeneratedStoryLine(
                        content=line,
                        requiresBranchInput=False,
                        branchPrompt=None,
                    )
                    for line in narrative_lines
                ),
                GeneratedStoryLine(
                    content=final_page.question,
                    requiresBranchInput=True,
                    branchPrompt=branch_prompt,
                ),
            ],
        )
        _log_generation_quality(
            operation="GENERATE",
            request=request,
            chapter=chapter,
        )
        return response

    async def continue_story(
        self,
        request: ContinueStoryRequest,
    ) -> GenerateStoryResponse:
        chapter_request = _build_chapter_request(request, conclude=True)
        chapter = await self._chapter_service.generate(chapter_request)
        chapter = chapter.validate_against_request(chapter_request)
        narrative_lines = _balanced_lines(
            _chapter_sentences(chapter),
            line_count=5,
        )
        response = GenerateStoryResponse(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            nextProgress=100,
            completed=True,
            lines=[
                GeneratedStoryLine(
                    content=line,
                    requiresBranchInput=False,
                    branchPrompt=None,
                )
                for line in narrative_lines
            ],
        )
        _log_generation_quality(
            operation="CONTINUE",
            request=request,
            chapter=chapter,
        )
        return response


def _build_chapter_request(
    request: GenerateStoryRequest,
    *,
    conclude: bool,
) -> StoryChapterGenerateRequest:
    title = request.storyTemplate.title.strip()
    context = request.storyTemplate.context.strip() or (
        f"{title}의 세계에서 주인공이 사건을 만나고 해결하는 이야기"
    )
    characters = _story_characters(title, context)
    history = list(request.history) if isinstance(request, ContinueStoryRequest) else []
    last_question = next(
        (
            line.content.strip()
            for line in reversed(history)
            if line.requiresBranchInput and line.content.strip()
        ),
        None,
    )
    recent_history = history[-4:]
    recent_pages = [
        {
            "pageNumber": index,
            "sentences": [line.content.strip()],
            "question": line.content.strip() if line.requiresBranchInput else None,
        }
        for index, line in enumerate(recent_history, start=1)
        if line.content.strip()
    ]
    rolling_summary = _rolling_summary(context, history)
    required_characters = [character["characterId"] for character in characters]
    if conclude:
        goal = "아이의 선택을 실제 사건으로 이어 받아 갈등을 풀고 이야기를 마무리한다."
        event_texts = (
            "아이의 선택이 첫 사건으로 바로 일어나고 주인공이 자연스럽게 반응한다",
            "그 선택이 만든 구체적인 결과와 짧은 대화를 보여 준다",
            "예상 밖의 마지막 어려움을 행동으로 해결한다",
            "앞선 사건을 회수하며 따뜻하고 분명한 결말을 맺는다",
        )
        question_focus = None
    else:
        goal = "이야기의 인물과 갈등을 흥미롭게 소개하고 다음 행동을 아이가 고르게 한다."
        event_texts = (
            "배경과 주인공의 성격이 드러나는 구체적인 사건이 시작된다",
            "주인공 사이의 차이나 바람이 자연스러운 대화로 드러난다",
            "바로 행동해야 하는 작은 문제나 놀라운 발견이 생긴다",
            "사건이 더 커지기 직전에 멈추고 다음 행동을 묻는다",
        )
        question_focus = "주인공이 바로 다음에 할 구체적이고 재미있는 행동 또는 말"
    payload = {
        "requestId": _chapter_request_id(request.requestId),
        "schemaVersion": 3,
        "storyId": request.storyId,
        "studentId": request.studentId,
        "storyRevision": 1 if conclude else 0,
        "chapterNumber": 2 if conclude else 1,
        "conclude": conclude,
        "storyTemplate": {
            "templateId": request.storyTemplate.storyTemplateId,
            "version": 1,
            "title": title,
            "context": context,
            "currentBeat": {
                "beatId": "legacy-ending" if conclude else "legacy-opening",
                "goal": goal,
                "questionFocus": question_focus,
                "allowedBranchSlots": [] if conclude else ["ACTION", "DIALOGUE"],
            },
        },
        "storyState": {
            "rollingSummary": rolling_summary,
            "resolvedFacts": [
                line.content.strip()
                for line in history
                if not line.requiresBranchInput and line.content.strip()
            ],
            "unresolvedHooks": [last_question] if last_question else [],
            "recentPages": recent_pages,
            "characters": characters,
            "lastQuestion": last_question,
        },
        "chapterPlan": {
            "orderedEvents": [
                {
                    "eventId": f"legacy-event-{index}",
                    "lockedEvent": event_text,
                    "requiredCharacters": required_characters,
                    "requiredConcepts": [title],
                }
                for index, event_text in enumerate(event_texts, start=1)
            ],
            "minPages": 2,
            "maxPages": 4,
            "questionFocus": question_focus,
        },
        "branchInput": (
            {
                "source": "TEXT_CONFIRMED",
                "text": request.branchIntent.strip(),
            }
            if isinstance(request, ContinueStoryRequest)
            else None
        ),
        "generationProfile": _balanced_generation_profile(
            [character["name"] for character in characters]
        ),
    }
    return StoryChapterGenerateRequest.model_validate(payload)


def _story_characters(title: str, context: str) -> list[dict[str, object]]:
    source = f"{title} {context}"
    matches = [
        (match.start(), name)
        for name in _KNOWN_CHARACTER_NAMES
        if (match := re.search(re.escape(name), source)) is not None
    ]
    names = list(dict.fromkeys(name for _, name in sorted(matches)))[:4]
    if "해와 바람" in source:
        names = list(dict.fromkeys(["해", "바람", *names]))[:4]
    if not names:
        names = ["주인공"]
    return [
        {
            "characterId": f"character_{index}",
            "name": name,
            "role": "이야기를 이끄는 주인공" if index == 1 else "함께 사건을 겪는 친구",
            "immutableTraits": ["이름과 기본 모습이 이야기 내내 유지됨"],
        }
        for index, name in enumerate(names, start=1)
    ]


def _balanced_generation_profile(protected_terms: list[str]) -> dict[str, object]:
    profile: dict[str, object] = {
        "schemaVersion": 2,
        "generationProfileVersion": 1,
        "sourceReadingProfileVersion": 1,
        "compilerVersion": "legacy-v1-balanced-v1",
        "contentContract": {
            "sentenceCount": 4,
            "preferredWrittenSyllables": {"min": 55, "max": 70},
            "acceptedWrittenSyllables": {"min": 50, "max": 75},
            "directDialogueCount": 1,
        },
        "skills": [
            {
                "code": "HAS_TENSE_ONSET",
                "role": "LIMITED",
                "maxOccurrences": 2,
                "unitPenalty": 1.2,
            },
            {
                "code": "PHONO_LIAISON",
                "role": "LIMITED",
                "maxOccurrences": 4,
                "unitPenalty": 1.4,
            },
            {
                "code": "HAS_COMPLEX_CODA",
                "role": "LIMITED",
                "maxOccurrences": 1,
                "unitPenalty": 1.4,
            },
        ],
        "protectedTerms": list(dict.fromkeys(protected_terms)),
    }
    canonical = json.dumps(
        profile,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    profile["policyHash"] = f"sha256:{digest}"
    return profile


def _rolling_summary(context: str, history: Sequence[object]) -> str:
    parts = [context]
    parts.extend(line.content.strip() for line in history if getattr(line, "content", "").strip())
    return " ".join(parts)[-4000:]


def _chapter_request_id(request_id: str) -> str:
    if len(request_id) <= 128:
        return request_id
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"legacy-{digest}"


def _chapter_sentences(chapter: StoryChapterGenerateResponse) -> list[str]:
    return [sentence for page in chapter.pages for sentence in page.sentences]


def _balanced_lines(sentences: Sequence[str], *, line_count: int) -> list[str]:
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(cleaned) < line_count:
        raise ValueError("generated chapter does not contain enough sentences")
    quotient, remainder = divmod(len(cleaned), line_count)
    lines: list[str] = []
    cursor = 0
    for index in range(line_count):
        size = quotient + (1 if index < remainder else 0)
        lines.append(" ".join(cleaned[cursor : cursor + size]))
        cursor += size
    return lines


def _unique_choice_labels(choices: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for index, choice in enumerate(choices, start=1):
        base = choice.strip()[:72] or f"{index}번 행동을 해요."
        label = base
        if label in labels:
            label = f"{base[:72]} {index}번"
        labels.append(label[:80])
    if len(labels) != 3 or len(set(labels)) != 3:
        raise ValueError("chapter choices must contain three distinct labels")
    return labels


def _quality_log_event(
    *,
    operation: str,
    request: GenerateStoryRequest,
    chapter: StoryChapterGenerateResponse,
) -> dict[str, object]:
    document = chapter.model_dump(by_alias=True, mode="json")
    return {
        "event": "story_generation_quality",
        "logSchemaVersion": 1,
        "operation": operation,
        "outcome": "SUCCESS",
        "requestId": request.requestId,
        "storyId": request.storyId,
        "generationId": document["generationId"],
        "chapterNumber": document["chapterNumber"],
        "quality": document["quality"],
        "generation": document["generation"],
        "timingMs": document["timingMs"],
    }


def _log_generation_quality(
    *,
    operation: str,
    request: GenerateStoryRequest,
    chapter: StoryChapterGenerateResponse,
) -> None:
    quality_logger.info(
        "%s",
        json.dumps(
            _quality_log_event(
                operation=operation,
                request=request,
                chapter=chapter,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


__all__ = ["LegacyStoryGenerationService", "StoryChapterGenerator"]
