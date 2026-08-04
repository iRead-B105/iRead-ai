from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import httpx

from iread_ai.personalization.prompts import (
    BASELINE_PROMPT_MODE,
    PERSONALIZED_PROMPT_MODE,
    build_reading_policy_hints,
)

logger = logging.getLogger(__name__)

DEFAULT_CHAPTER_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "chapter_personalized.md"
)
DEFAULT_BASELINE_CHAPTER_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "chapter_baseline.md"
)
CHAPTER_PROMPT_MODES = frozenset({BASELINE_PROMPT_MODE, PERSONALIZED_PROMPT_MODE})
ChapterPromptMode = Literal["baseline", "personalized"]


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


@dataclass(frozen=True, slots=True)
class ChapterGenerationContext:
    story_title: str
    story_context: str
    chapter_goal: str
    ordered_events: tuple[str, ...]
    chapter_number: int = 1
    child_input: str = ""
    last_question: str | None = None
    previous_context: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()
    question_focus: str | None = None
    conclude: bool = False
    expected_page_count: int = 3
    expected_sentences_per_page: int = 3

    def __post_init__(self) -> None:
        _require_nonblank(self.story_title, "story_title")
        _require_nonblank(self.story_context, "story_context")
        _require_nonblank(self.chapter_goal, "chapter_goal")
        if self.chapter_number < 1:
            raise ValueError("chapter_number must be at least one")
        if not 2 <= self.expected_page_count <= 4:
            raise ValueError("expected_page_count must be between 2 and 4")
        if self.expected_sentences_per_page not in {3, 4}:
            raise ValueError("expected_sentences_per_page must be 3 or 4")
        if not self.ordered_events:
            raise ValueError("ordered_events must contain at least one event")
        for field_name, values in (
            ("ordered_events", self.ordered_events),
            ("previous_context", self.previous_context),
            ("characters", self.characters),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} must contain only nonblank strings")
        if self.conclude:
            if self.question_focus is not None:
                raise ValueError("question_focus is not allowed for a concluding chapter")
        else:
            if self.question_focus is None:
                raise ValueError("question_focus is required for a continuing chapter")
            _require_nonblank(self.question_focus, "question_focus")
        if self.last_question is not None:
            _require_nonblank(self.last_question, "last_question")

    def to_dict(self) -> dict[str, Any]:
        child_branch_active = bool(self.child_input.strip())
        spoken_answer = bool(
            child_branch_active
            and self.last_question is not None
            and any(
                marker in self.last_question
                for marker in (
                    "구호",
                    "노래",
                    "대답",
                    "리듬",
                    "말",
                    "약속",
                    "외치",
                    "응원",
                    "인사",
                )
            )
        )
        answer_owner_candidates = (
            [
                character.strip()
                for character in self.characters
                if self.last_question is not None and character.strip() in self.last_question
            ]
            if child_branch_active
            else []
        )
        return {
            "story_title": self.story_title.strip(),
            "story_context": self.story_context.strip(),
            "chapter_number": self.chapter_number,
            "chapter_goal": self.chapter_goal.strip(),
            "ordered_events": [event.strip() for event in self.ordered_events],
            "child_input": self.child_input.strip(),
            "last_question": (
                self.last_question.strip() if self.last_question is not None else None
            ),
            "previous_context": [context.strip() for context in self.previous_context],
            "characters": [character.strip() for character in self.characters],
            "question_focus": (
                self.question_focus.strip() if self.question_focus is not None else None
            ),
            "chapter_mode": "ending" if self.conclude else "continuing",
            "expected_page_count": self.expected_page_count,
            "expected_sentence_count": (
                self.expected_page_count * self.expected_sentences_per_page
            ),
            "child_branch_plan": {
                "active": child_branch_active,
                "answer_to_question": (
                    self.last_question.strip()
                    if child_branch_active and self.last_question is not None
                    else None
                ),
                "answer_owner_candidates": answer_owner_candidates,
                "required_literal_signal": (
                    self.child_input.strip() if child_branch_active else None
                ),
                "answer_delivery": (
                    "natural_direct_dialogue" if spoken_answer else "natural_narrative_event"
                )
                if child_branch_active
                else None,
                "forbidden_story_placeholders": (
                    [
                        "직전 답",
                        "직전 선택",
                        "직전 방법",
                        "확정된 답",
                        "아이 답",
                    ]
                    if child_branch_active
                    else []
                ),
                "sentence_roles": (
                    [
                        (
                            "첫 3문장: 답을 기존 인물의 자연스러운 대사와 행동으로 실행"
                            if spoken_answer
                            else "첫 3문장: 아이 답이 이야기 속 실제 사건으로 발생"
                        ),
                        "첫 3문장: 기존 등장인물이 구체적으로 반응",
                        "첫 3문장: 답의 결과를 본래 장 사건의 원인으로 연결",
                    ]
                    if child_branch_active
                    else []
                ),
                "end_sentence_index": 3 if child_branch_active else None,
                "must_return_to_main_story": child_branch_active,
                "later_callback_required": child_branch_active,
            },
        }


@dataclass(frozen=True, slots=True)
class ChapterCandidate:
    candidate_id: str
    sentences: tuple[str, ...]
    child_detour_end_sentence_index: int | None = None
    question: str | None = None
    subtitle: str | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonblank(self.candidate_id, "candidate_id")
        if not 8 <= len(self.sentences) <= 16:
            raise ValueError("a chapter candidate must contain between 8 and 16 sentences")
        for sentence in self.sentences:
            _require_nonblank(sentence, "sentence")
            if not any("가" <= character <= "힣" for character in sentence):
                raise ValueError("each sentence must contain Korean text")
        detour_end = self.child_detour_end_sentence_index
        if detour_end is not None:
            if detour_end not in {3, 4}:
                raise ValueError("child_detour_end_sentence_index must be 3 or 4")
            if detour_end > len(self.sentences):
                raise ValueError("child_detour_end_sentence_index exceeds sentence count")
        if self.question is None:
            if self.subtitle is not None or self.choices:
                raise ValueError("subtitle and choices require a question")
        else:
            _require_nonblank(self.question, "question")
            if self.subtitle is not None:
                _require_nonblank(self.subtitle, "subtitle")
                if len(self.subtitle) > 40:
                    raise ValueError("subtitle must contain at most 40 characters")
            if len(self.choices) != 3:
                raise ValueError("a continuing chapter must contain exactly three choices")
            for choice in self.choices:
                _require_nonblank(choice, "choice")
            if len({choice.strip() for choice in self.choices}) != 3:
                raise ValueError("chapter choices must be distinct")

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "candidateId": self.candidate_id,
            "sentences": list(self.sentences),
            "childDetourEndSentenceIndex": (self.child_detour_end_sentence_index),
        }
        if self.question is not None:
            document["question"] = self.question
            document["subtitle"] = self.subtitle
            document["choices"] = list(self.choices)
        return document


@dataclass(frozen=True, slots=True)
class ChapterGenerationBatch:
    candidates: tuple[ChapterCandidate, ...]
    raw_output: str
    elapsed_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    system_prompt: str = ""
    user_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("chapter generation batch must contain at least one candidate")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must not be negative")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "candidateCount": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rawOutput": self.raw_output,
            "elapsedMs": round(self.elapsed_ms, 3),
            "usage": dict(self.usage),
            "systemPrompt": self.system_prompt,
            "userPrompt": self.user_prompt,
        }


@runtime_checkable
class ChapterCandidateGenerator(Protocol):
    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: Any | None,
        *,
        candidate_count: int = 3,
        prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
    ) -> ChapterGenerationBatch: ...


class MockChapterCandidateGenerator:
    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: Any | None,
        *,
        candidate_count: int = 3,
        prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
    ) -> ChapterGenerationBatch:
        _validate_generation_options(
            profile=profile,
            candidate_count=candidate_count,
            prompt_mode=prompt_mode,
        )
        started = time.perf_counter()
        system_prompt = load_chapter_prompt(prompt_mode=prompt_mode)
        user_prompt = build_chapter_user_prompt(
            context=context,
            profile=profile,
            prompt_mode=prompt_mode,
        )
        candidates = tuple(
            _mock_candidate(context, candidate_number)
            for candidate_number in range(1, candidate_count + 1)
        )
        raw_output = json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "sentences": list(candidate.sentences),
                        "child_detour_end_sentence_index": (
                            candidate.child_detour_end_sentence_index
                        ),
                        **(
                            {
                                "question": candidate.question,
                                "choices": list(candidate.choices),
                            }
                            if candidate.question is not None
                            else {}
                        ),
                    }
                    for candidate in candidates
                ]
            },
            ensure_ascii=False,
        )
        return ChapterGenerationBatch(
            candidates=candidates,
            raw_output=raw_output,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            usage={},
            model="mock",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


class OpenAIChapterCandidateGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 32.0,
        max_output_tokens: int = 4200,
        client: httpx.AsyncClient | None = None,
        prompt_path: Path | None = None,
        baseline_prompt_path: Path | None = None,
    ) -> None:
        _require_nonblank(api_key, "api_key")
        _require_nonblank(model, "model")
        _require_nonblank(base_url, "base_url")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens < 512:
            raise ValueError("max_output_tokens must be at least 512")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._client = client
        self._prompt_path = prompt_path
        self._baseline_prompt_path = baseline_prompt_path

    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: Any | None,
        *,
        candidate_count: int = 3,
        prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
    ) -> ChapterGenerationBatch:
        _validate_generation_options(
            profile=profile,
            candidate_count=candidate_count,
            prompt_mode=prompt_mode,
        )
        system_prompt = load_chapter_prompt(
            (
                self._baseline_prompt_path
                if prompt_mode == BASELINE_PROMPT_MODE
                else self._prompt_path
            ),
            prompt_mode=prompt_mode,
        )
        user_prompt = build_chapter_user_prompt(
            context=context,
            profile=profile,
            prompt_mode=prompt_mode,
        )
        payload = {
            "model": self._model,
            "store": False,
            "reasoning": {"effort": "low"},
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "iread_personalized_story_chapter",
                    "strict": True,
                    "schema": _chapter_candidate_schema(
                        candidate_count,
                        child_branch_active=bool(context.child_input.strip()),
                        branch_required=not context.conclude,
                        expected_page_count=context.expected_page_count,
                        expected_sentence_count=(
                            context.expected_page_count * context.expected_sentences_per_page
                        ),
                    ),
                }
            },
            "max_output_tokens": self._max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        for attempt in range(2):
            try:
                response = await self._post(payload=payload, headers=headers)
            except ChapterGenerationError as exc:
                if attempt == 0 and exc.retryable:
                    continue
                raise
            if response.status_code >= 400:
                raise ChapterGenerationError(
                    (f"OpenAI Responses API request failed with status {response.status_code}"),
                    retryable=(
                        response.status_code in {408, 409, 429} or response.status_code >= 500
                    ),
                )

            try:
                data = response.json()
                if data.get("status") not in {None, "completed"}:
                    raise ValueError("response did not complete")
                raw_output = _extract_output_text(data)
                document = json.loads(raw_output)
                candidates = _parse_candidates(
                    document,
                    expected_count=candidate_count,
                    child_branch_active=bool(context.child_input.strip()),
                    branch_required=not context.conclude,
                )
                usage = data.get("usage", {})
                if not isinstance(usage, dict):
                    usage = {}
                break
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    continue
                raise ChapterGenerationError(
                    (
                        "OpenAI returned an invalid story-chapter document "
                        f"({type(exc).__name__}: {exc})"
                    ),
                    retryable=False,
                    raw_output=locals().get("raw_output"),
                ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        return ChapterGenerationBatch(
            candidates=candidates,
            raw_output=raw_output,
            elapsed_ms=elapsed_ms,
            usage=cast(dict[str, Any], usage),
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def _post(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                if self._client is not None:
                    return await self._client.post(
                        f"{self._base_url}/responses",
                        headers=headers,
                        json=payload,
                        timeout=self._timeout_seconds,
                    )
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    return await client.post(
                        f"{self._base_url}/responses",
                        headers=headers,
                        json=payload,
                    )
        except (TimeoutError, httpx.TimeoutException) as exc:
            logger.warning(
                "Story chapter request timed out exception_type=%s",
                type(exc).__name__,
            )
            raise ChapterGenerationError(
                "story-chapter generation timed out",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "Story chapter request failed exception_type=%s",
                type(exc).__name__,
            )
            raise ChapterGenerationError(
                "story-chapter model is unavailable",
                retryable=True,
            ) from exc


class ChapterGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.raw_output = raw_output


def load_chapter_prompt(
    path: Path | None = None,
    *,
    prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
) -> str:
    if prompt_mode not in CHAPTER_PROMPT_MODES:
        raise ValueError(f"unsupported chapter prompt mode: {prompt_mode}")
    prompt_path = path or (
        DEFAULT_BASELINE_CHAPTER_PROMPT_PATH
        if prompt_mode == BASELINE_PROMPT_MODE
        else DEFAULT_CHAPTER_PROMPT_PATH
    )
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("chapter prompt must not be empty")
    if len(prompt.encode("utf-8")) > 64 * 1024:
        raise ValueError("chapter prompt must not exceed 64 KiB")
    return prompt


def build_chapter_user_prompt(
    *,
    context: ChapterGenerationContext,
    profile: Any | None,
    prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
) -> str:
    if prompt_mode not in CHAPTER_PROMPT_MODES:
        raise ValueError(f"unsupported chapter prompt mode: {prompt_mode}")
    if prompt_mode == PERSONALIZED_PROMPT_MODE and profile is None:
        raise ValueError("personalized chapter generation requires a generation profile")
    document: dict[str, Any] = {
        "chapter": context.to_dict(),
    }
    if prompt_mode == PERSONALIZED_PROMPT_MODE:
        document["generation_profile"] = _jsonable(profile)
        document["reading_policy_hints"] = build_reading_policy_hints(profile)
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_generation_options(
    *,
    profile: Any | None,
    candidate_count: int,
    prompt_mode: ChapterPromptMode,
) -> None:
    if prompt_mode not in CHAPTER_PROMPT_MODES:
        raise ValueError(f"unsupported chapter prompt mode: {prompt_mode}")
    if prompt_mode == PERSONALIZED_PROMPT_MODE and profile is None:
        raise ValueError("personalized chapter generation requires a generation profile")
    if not 1 <= candidate_count <= 8:
        raise ValueError("candidate_count must be between 1 and 8")


def _chapter_candidate_schema(
    candidate_count: int,
    *,
    child_branch_active: bool,
    branch_required: bool,
    expected_page_count: int,
    expected_sentence_count: int,
) -> dict[str, Any]:
    candidate_required = [
        "candidate_id",
        "sentences",
        "child_detour_end_sentence_index",
        "dialogue_sentence_indexes",
        "child_input_later_effect",
        "question_open_slot",
    ]
    candidate_properties: dict[str, Any] = {
        "candidate_id": {
            "type": "string",
            "minLength": 1,
        },
        "sentences": {
            "type": "array",
            "minItems": expected_sentence_count,
            "maxItems": expected_sentence_count,
            "items": {
                "type": "string",
                "minLength": 1,
                "pattern": "[가-힣]",
            },
        },
        "child_detour_end_sentence_index": (
            {
                "type": "integer",
                "enum": [3],
            }
            if child_branch_active
            else {"type": "null"}
        ),
        "dialogue_sentence_indexes": {
            "type": "array",
            "minItems": (expected_page_count + 1) // 2,
            "maxItems": expected_page_count,
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": expected_sentence_count,
            },
        },
        "child_input_later_effect": (
            {
                "type": "string",
                "minLength": 1,
            }
            if child_branch_active
            else {"type": "null"}
        ),
        "question_open_slot": (
            {
                "type": "string",
                "minLength": 1,
            }
            if branch_required
            else {"type": "null"}
        ),
    }
    if branch_required:
        candidate_required.extend(("question", "subtitle", "choices"))
        candidate_properties.update(
            {
                "question": {
                    "type": "string",
                    "minLength": 1,
                },
                "subtitle": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 40,
                },
                "choices": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": candidate_required,
                    "properties": candidate_properties,
                },
            }
        },
    }


def _parse_candidates(
    document: dict[str, Any],
    *,
    expected_count: int,
    child_branch_active: bool,
    branch_required: bool,
) -> tuple[ChapterCandidate, ...]:
    raw_candidates = document["candidates"]
    if not isinstance(raw_candidates, list) or (len(raw_candidates) != expected_count):
        raise ValueError("candidate count does not match the request")

    parsed: list[ChapterCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise TypeError("candidate must be an object")
        raw_sentences = raw_candidate["sentences"]
        if not isinstance(raw_sentences, list):
            raise TypeError("sentences must be an array")
        raw_detour_end = raw_candidate["child_detour_end_sentence_index"]
        detour_end = int(raw_detour_end) if raw_detour_end is not None else None
        if child_branch_active and detour_end is None:
            raise ValueError("child input requires child_detour_end_sentence_index")
        if not child_branch_active and detour_end is not None:
            raise ValueError("child_detour_end_sentence_index requires child input")
        raw_choices = raw_candidate.get("choices", [])
        if branch_required and not isinstance(raw_choices, list):
            raise TypeError("choices must be an array")
        parsed.append(
            ChapterCandidate(
                candidate_id=str(raw_candidate["candidate_id"]).strip(),
                sentences=tuple(str(sentence).strip() for sentence in raw_sentences),
                child_detour_end_sentence_index=detour_end,
                question=(str(raw_candidate["question"]).strip() if branch_required else None),
                subtitle=(str(raw_candidate["subtitle"]).strip() if branch_required else None),
                choices=(
                    tuple(str(choice).strip() for choice in raw_choices) if branch_required else ()
                ),
            )
        )
    candidate_ids = [candidate.candidate_id for candidate in parsed]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ids must be unique")
    return tuple(parsed)


def _extract_output_text(data: dict[str, Any]) -> str:
    convenience_text = data.get("output_text")
    if isinstance(convenience_text, str) and convenience_text:
        return convenience_text

    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif content.get("type") == "refusal":
                raise ValueError("model refused the story-chapter request")
    output_text = "".join(chunks)
    if not output_text:
        raise ValueError("response contained no output text")
    return output_text


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", by_alias=True))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _mock_candidate(
    context: ChapterGenerationContext,
    candidate_number: int,
) -> ChapterCandidate:
    main_character = context.characters[0] if context.characters else "친구"
    companion = context.characters[1] if len(context.characters) > 1 else "작은 친구"
    child_input = context.child_input.strip()
    planned_sentences = _mock_planned_sentences(
        context,
        main_character=main_character,
        companion=companion,
    )
    if child_input:
        short_input = _truncate_hangul(child_input, max_syllables=8)
        sentences = (
            f"아이의 {short_input} 선택이 곧 이루어져요.",
            f"{main_character}는 그 선택을 믿고 움직여요.",
            f"{companion}도 곁에서 힘껏 도와주어요.",
            *planned_sentences[:9],
        )
        detour_end = 3
    else:
        sentences = planned_sentences
        detour_end = None
    suffix = "다음 장면에서 어떤 일이 이어지면 좋을까요?"
    return ChapterCandidate(
        candidate_id=f"candidate-{candidate_number}",
        sentences=sentences,
        child_detour_end_sentence_index=detour_end,
        question=suffix if not context.conclude else None,
        subtitle="다음 모험의 갈림길" if not context.conclude else None,
        choices=(
            (
                "함께 한 걸음 더 가요.",
                "잠깐 멈춰 살펴봐요.",
                "다른 길을 찾아봐요.",
            )
            if not context.conclude
            else ()
        ),
    )


def _mock_planned_sentences(
    context: ChapterGenerationContext,
    *,
    main_character: str,
    companion: str,
) -> tuple[str, ...]:
    reactions = (
        "두 친구는 서로의 생각을 차분히 나눠요.",
        "새로운 흔적이 숲길 위로 선명히 보여요.",
        "작은 선택이 다음 사건을 크게 바꾸어요.",
        "남은 궁금증이 다음 모험으로 이어져요.",
    )
    sentences: list[str] = []
    for index, raw_event in enumerate(context.ordered_events[:4]):
        event = raw_event.strip()
        if event.startswith("사건: "):
            event = event.removeprefix("사건: ")
        event = event.split(". 반드시 포함할 개념:", maxsplit=1)[0].strip()
        if event and event[-1] not in ".!?":
            event += "."
        short_event = _truncate_hangul(event, max_syllables=12)
        sentences.extend(
            (
                f"{short_event} 일이 시작돼요.",
                reactions[index],
                "친구는 “다음 단서를 찾아보자!” 하고 말해요.",
            )
        )

    fallback = (
        _mock_sentence(context.chapter_goal),
        f"{main_character}는 이야기의 목표를 향해 움직여요.",
        f"{companion}도 자기 몫의 행동을 이어 가요.",
        f"{main_character}와 {companion}는 다음 사건을 함께 맞아요.",
    )
    for sentence in fallback:
        if len(sentences) >= 12:
            break
        sentences.append(sentence)
    while len(sentences) < 12:
        sentences.append(reactions[len(sentences) % len(reactions)])
    return tuple(sentences[:12])


def _mock_sentence(text: str) -> str:
    sentence = text.strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _truncate_hangul(text: str, *, max_syllables: int) -> str:
    result: list[str] = []
    syllables = 0
    for character in text.strip().rstrip(".?!。？！ "):
        if "가" <= character <= "힣":
            if syllables == max_syllables:
                break
            syllables += 1
        result.append(character)
    return "".join(result).strip()


__all__ = [
    "CHAPTER_PROMPT_MODES",
    "ChapterCandidate",
    "ChapterCandidateGenerator",
    "ChapterGenerationBatch",
    "ChapterGenerationContext",
    "ChapterGenerationError",
    "ChapterPromptMode",
    "DEFAULT_BASELINE_CHAPTER_PROMPT_PATH",
    "DEFAULT_CHAPTER_PROMPT_PATH",
    "MockChapterCandidateGenerator",
    "OpenAIChapterCandidateGenerator",
    "build_chapter_user_prompt",
    "load_chapter_prompt",
]
