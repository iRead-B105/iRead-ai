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
_LEGACY_GENERATION_ATTEMPTS = 3


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
    "노인",
    "소년",
    "청새치",
    "상어",
    "신데렐라",
    "계모",
    "요정",
    "자라",
    "용왕",
)


class LegacyStoryGenerationService:
    def __init__(self, *, chapter_service: StoryChapterGenerator) -> None:
        self._chapter_service = chapter_service

    async def generate(
        self,
        request: GenerateStoryRequest,
    ) -> GenerateStoryResponse:
        if request.currentProgress != 0:
            raise ValueError("initial story generation requires currentProgress=0")
        chapter_request = _build_chapter_request(
            request,
            conclude=False,
            narrative_count=3,
        )
        for attempt in range(_LEGACY_GENERATION_ATTEMPTS):
            chapter = await self._chapter_service.generate(chapter_request)
            chapter = chapter.validate_against_request(chapter_request)
            final_page = chapter.pages[-1]
            try:
                narrative_lines = _three_sentence_lines(
                    _chapter_sentences(chapter),
                    line_count=3,
                )
                if final_page.question is None or len(final_page.choices) != 3:
                    raise ValueError("opening chapter must end with one question and three choices")
                _validate_branch_question(final_page.question)
                break
            except ValueError:
                if attempt == _LEGACY_GENERATION_ATTEMPTS - 1:
                    raise
        branch_prompt = StoryBranchPrompt(
            subtitle=(
                getattr(final_page, "subtitle", None)
                or _branch_subtitle(final_page.question, final_page.choices)
            ),
            options=[
                StoryBranchOption(
                    optionNo=index,
                    label=label,
                )
                for index, label in enumerate(
                    _unique_choice_labels(final_page.choices),
                    start=1,
                )
            ],
        )
        response = GenerateStoryResponse(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            nextProgress=4,
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
        page_count = len(request.history)
        page_in_day = page_count % 10
        if request.currentProgress != page_count:
            raise ValueError("currentProgress must match history page count")
        if page_in_day == 4:
            narrative_count = 4
            requires_branch = True
        elif page_in_day == 9:
            narrative_count = 1
            requires_branch = False
        elif page_in_day == 0 and 0 < page_count < 100:
            narrative_count = 3
            requires_branch = True
        else:
            raise ValueError("story continuation must start after page 4, 9, or 10")

        chapter_request = _build_chapter_request(
            request,
            conclude=not requires_branch,
            narrative_count=narrative_count,
        )
        for attempt in range(_LEGACY_GENERATION_ATTEMPTS):
            chapter = await self._chapter_service.generate(chapter_request)
            chapter = chapter.validate_against_request(chapter_request)
            chapter_sentences = _chapter_sentences(chapter)
            try:
                narrative_lines = _three_sentence_lines(
                    chapter_sentences,
                    line_count=narrative_count,
                )
                final_page = chapter.pages[-1]
                if requires_branch and (
                    final_page.question is None or len(final_page.choices) != 3
                ):
                    raise ValueError(
                        "branching chapter must end with one question and three choices"
                    )
                if requires_branch:
                    previous_question = next(
                        (
                            line.content
                            for line in reversed(request.history)
                            if line.requiresBranchInput
                        ),
                        None,
                    )
                    _validate_branch_question(
                        final_page.question,
                        previous_question=previous_question,
                    )
                break
            except ValueError:
                if attempt == _LEGACY_GENERATION_ATTEMPTS - 1:
                    raise
        lines = [
            GeneratedStoryLine(
                content=line,
                requiresBranchInput=False,
                branchPrompt=None,
            )
            for line in narrative_lines
        ]
        if requires_branch:
            final_page = chapter.pages[-1]
            lines.append(
                GeneratedStoryLine(
                    content=final_page.question,
                    requiresBranchInput=True,
                    branchPrompt=StoryBranchPrompt(
                        subtitle=(
                            getattr(final_page, "subtitle", None)
                            or _branch_subtitle(
                                final_page.question,
                                final_page.choices,
                            )
                        ),
                        options=[
                            StoryBranchOption(optionNo=index, label=label)
                            for index, label in enumerate(
                                _unique_choice_labels(final_page.choices),
                                start=1,
                            )
                        ],
                    ),
                )
            )
        next_progress = page_count + len(lines)
        response = GenerateStoryResponse(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            nextProgress=next_progress,
            completed=next_progress == 100,
            lines=lines,
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
    narrative_count: int,
) -> StoryChapterGenerateRequest:
    title = request.storyTemplate.title.strip()
    context = request.storyTemplate.context.strip() or (
        f"{title}의 세계에서 주인공이 사건을 만나고 해결하는 이야기"
    )
    characters = _story_characters(title, context)
    history = list(request.history) if isinstance(request, ContinueStoryRequest) else []
    page_count = len(history)
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
    final_story_page = page_count >= 99
    arc_instruction = _arc_instruction(page_count)
    if conclude and final_story_page:
        goal = (
            f"{arc_instruction} 아이의 마지막 선택과 앞선 복선을 회수해 "
            "핵심 갈등을 풀고 명확한 결말을 맺는다."
        )
        event_texts = (
            "아이의 마지막 선택이 실제 사건으로 일어난다",
            "선택의 결과로 핵심 갈등이 해결된다",
            "앞선 약속과 단서를 회수한다",
            "인물의 변화가 드러나는 따뜻한 결말을 맺는다",
        )
        question_focus = None
    elif conclude:
        goal = (
            f"{arc_instruction} 아이의 선택이 만든 결과로 오늘의 장면만 닫되 "
            "핵심 갈등과 다음 날의 궁금증은 유지한다."
        )
        event_texts = (
            "아이의 선택이 당일 마지막 사건으로 일어난다",
            "등장인물이 그 결과에 구체적으로 반응한다",
            "오늘의 작은 목표는 마무리한다",
            "핵심 갈등을 닫지 않고 다음 날 이어질 단서를 남긴다",
        )
        question_focus = None
    else:
        goal = (
            f"{arc_instruction} 지금까지의 선택이 만든 인과관계를 이어 가고 "
            "현재 장면에서 실행 가능한 다음 행동을 아이가 고르게 한다."
        )
        if isinstance(request, ContinueStoryRequest):
            branch_intent = request.branchIntent.strip()
            event_texts = (
                f"확정된 선택 '{branch_intent}'이 기존 인물의 행동과 사건으로 일어난다",
                "그 선택 때문에 주변 상황과 인물의 행동이 눈에 보이게 달라진다",
                "선택의 결과 뒤에 이전과 다른 작은 문제나 발견이 생긴다",
                "직전 선택은 되묻지 않고 새 문제의 다음 행동을 물을 직전에 멈춘다",
            )
            question_focus = (
                f"직전 질문 '{last_question or '없음'}'은 이미 답이 끝났다. "
                f"확정된 선택 '{branch_intent}'의 결과 뒤 새로 생긴 문제에서 "
                "주인공이 다음에 할 구체적인 행동 또는 말. 직전 선택 대상은 재질문 금지"
            )
        else:
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
        "storyRevision": page_count,
        "chapterNumber": page_count // 10 + 1,
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
            "minPages": 3 if narrative_count == 1 else narrative_count,
            "maxPages": 3 if narrative_count == 1 else narrative_count,
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
            "sentenceCount": 3,
            "preferredWrittenSyllables": {"min": 39, "max": 57},
            "acceptedWrittenSyllables": {"min": 30, "max": 66},
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


def _three_sentence_lines(sentences: Sequence[str], *, line_count: int) -> list[str]:
    cleaned = [
        _ensure_sentence_ending(sentence.strip()) for sentence in sentences if sentence.strip()
    ]
    target_count = line_count * 3
    if len(cleaned) < target_count:
        raise ValueError("generated chapter does not contain enough sentences")
    if len(cleaned) == target_count:
        selected = cleaned
    else:
        selected = [
            cleaned[(index * (len(cleaned) - 1)) // (target_count - 1)]
            for index in range(target_count)
        ]
    lines: list[str] = []
    for index in range(line_count):
        page_sentences = selected[index * 3 : (index + 1) * 3]
        _validate_page_sentences(page_sentences)
        lines.append(" ".join(page_sentences))
    return lines


def _ensure_sentence_ending(sentence: str) -> str:
    return sentence if sentence.endswith((".", "!", "?", "。", "！", "？")) else sentence + "."


def _hangul_count(value: str) -> int:
    return sum(1 for character in value if "가" <= character <= "힣")


def _validate_page_sentences(sentences: Sequence[str]) -> None:
    if len(sentences) != 3:
        raise ValueError("each story page must contain exactly three sentences")
    for sentence in sentences:
        syllable_count = _hangul_count(sentence)
        if syllable_count < 10 or syllable_count > 22:
            raise ValueError("each story sentence must contain 10 to 22 Hangul syllables")
        if sentence.rstrip().endswith(("?", "？")):
            raise ValueError("story body must not contain a reader-facing branch question")
    normalized = [re.sub(r"[^가-힣 ]", "", sentence).strip() for sentence in sentences]
    first_words = [sentence.split(maxsplit=1)[0] for sentence in normalized if sentence]
    final_words = [sentence.rsplit(maxsplit=1)[-1] for sentence in normalized if sentence]
    if len(first_words) == 3 and len(set(first_words)) == 1:
        raise ValueError("story page sentences must not all begin with the same word")
    if len(final_words) == 3 and len(set(final_words)) == 1:
        raise ValueError("story page sentences must not all end with the same word")
    filler_phrases = (
        "장면이 이어져요",
        "이야기가 이어져요",
        "새로운 길이 천천히 열려요",
        "그 모습이 또렷해요",
    )
    if any(phrase in sentence for phrase in filler_phrases for sentence in sentences):
        raise ValueError("story page must not use mechanical filler phrases")


def _validate_branch_question(
    question: str,
    *,
    previous_question: str | None = None,
) -> None:
    normalized = re.sub(r"[\s.?!。？！]+", "", question)
    if not normalized:
        raise ValueError("branch question must not be blank")
    if previous_question is None:
        return
    previous_normalized = re.sub(r"[\s.?!。？！]+", "", previous_question)
    if normalized == previous_normalized:
        raise ValueError("branch question must advance beyond the previous question")


def _branch_subtitle(question: str, choices: Sequence[str]) -> str:
    source = next((choice.strip() for choice in choices if choice.strip()), question.strip())
    source = re.sub(r"[.?!。？！]+$", "", source)
    source = re.sub(r"\s+", " ", source).strip()
    if not source:
        raise ValueError("branch subtitle source must not be empty")
    return source[:40]


def _arc_instruction(page_count: int) -> str:
    next_page = min(page_count + 1, 100)
    if next_page <= 25:
        return "기 단계에서 인물, 목표와 핵심 갈등을 구체적인 사건으로 세운다."
    if next_page <= 50:
        return "승 단계에서 시도와 장애를 늘리고 이전 선택의 결과를 누적한다."
    if next_page <= 75:
        return "전 단계에서 예상 밖의 전환과 위기를 만들고 선택의 대가를 드러낸다."
    if next_page < 100:
        return "결 단계에서 핵심 갈등을 해결해 가며 앞선 단서와 약속을 회수한다."
    return "100번째 페이지에서 누적된 선택과 복선을 회수해 결말을 완성한다."


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
