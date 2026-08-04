from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

import httpx

from iread_ai.personalization.prompts import (
    build_repair_user_prompt,
    load_repair_prompt,
)

RepairStatus = Literal["REPAIRED", "UNABLE"]


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


@dataclass(frozen=True, slots=True)
class PageGenerationContext:
    story_title: str = "함께 만드는 이야기"
    story_context: str = ""
    locked_event: str = ""
    page_number: int = 1
    child_input: str = ""
    previous_pages: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()
    required_concepts: tuple[str, ...] = ()
    question_focus: str | None = None
    conclude: bool = False
    expected_sentence_count: int = 4

    def __post_init__(self) -> None:
        _require_nonblank(self.story_title, "story_title")
        _require_nonblank(self.story_context, "story_context")
        _require_nonblank(self.locked_event, "locked_event")
        if self.page_number not in {1, 2, 3, 4}:
            raise ValueError("page_number must be between 1 and 4")
        if any(not isinstance(item, str) or not item.strip() for item in self.previous_pages):
            raise ValueError("previous_pages must contain only nonblank strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.characters):
            raise ValueError("characters must contain only nonblank strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.required_concepts):
            raise ValueError("required_concepts must contain only nonblank strings")
        if self.question_focus is not None:
            _require_nonblank(self.question_focus, "question_focus")
        if self.expected_sentence_count not in {3, 4}:
            raise ValueError("expected_sentence_count must be 3 or 4")

    def to_dict(self) -> dict[str, Any]:
        branch_page_active = self.page_number == 1 and bool(self.child_input.strip())
        return {
            "story_title": self.story_title.strip(),
            "story_context": self.story_context.strip(),
            "locked_event": self.locked_event.strip(),
            "page_number": self.page_number,
            "child_input": self.child_input.strip(),
            "previous_pages": list(self.previous_pages),
            "characters": list(self.characters),
            "required_concepts": list(self.required_concepts),
            "question_focus": (
                self.question_focus.strip() if self.question_focus is not None else None
            ),
            "expected_sentence_count": self.expected_sentence_count,
            "branch_page_plan": {
                "active": branch_page_active,
                "sentence_roles": (
                    [
                        "1문장: 아이 답이 이야기 속 실제 사건으로 발생",
                        "2문장: 기존 등장인물이 그 사건에 구체적으로 반응",
                        "3문장: 엉뚱하고 짧은 결과가 한 번 더 이어짐",
                        ("4문장: 재미의 흔적은 남기되 locked_event의 본 흐름으로 복귀"),
                    ]
                    if branch_page_active
                    else []
                ),
                "return_event": (self.locked_event.strip() if branch_page_active else None),
            },
            "chapter_mode": "ending" if self.conclude else "continuing",
        }


@dataclass(frozen=True, slots=True)
class PageCandidate:
    candidate_id: str
    sentences: tuple[str, ...]
    question: str | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonblank(self.candidate_id, "candidate_id")
        if len(self.sentences) not in {3, 4}:
            raise ValueError("a page candidate must contain 3 or 4 sentences")
        for sentence in self.sentences:
            _require_nonblank(sentence, "sentence")
            if not any("가" <= character <= "힣" for character in sentence):
                raise ValueError("each sentence must contain Korean text")
        if self.question is None:
            if self.choices:
                raise ValueError("choices require a question")
        else:
            _require_nonblank(self.question, "question")
            if len(self.choices) != 3:
                raise ValueError("a branch question must contain exactly three choices")
            for choice in self.choices:
                _require_nonblank(choice, "choice")
            if len({choice.strip() for choice in self.choices}) != 3:
                raise ValueError("page choices must be distinct")

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "candidateId": self.candidate_id,
            "sentences": list(self.sentences),
        }
        if self.question is not None:
            document["question"] = self.question
            document["choices"] = list(self.choices)
        return document


@dataclass(frozen=True, slots=True)
class RepairReplacement:
    sentence_index: int
    sentence: str

    def __post_init__(self) -> None:
        if self.sentence_index not in {1, 2, 3, 4}:
            raise ValueError("sentence_index must be between 1 and 4")
        _require_nonblank(self.sentence, "sentence")
        if not any("가" <= character <= "힣" for character in self.sentence):
            raise ValueError("replacement sentence must contain Korean text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentenceIndex": self.sentence_index,
            "sentence": self.sentence,
        }


@dataclass(frozen=True, slots=True)
class RepairBatch:
    source_candidate_id: str
    repair_status: RepairStatus
    replacements: tuple[RepairReplacement, ...]
    raw_output: str
    elapsed_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    system_prompt: str = ""
    user_prompt: str = ""

    def __post_init__(self) -> None:
        _require_nonblank(self.source_candidate_id, "source_candidate_id")
        if self.repair_status not in {"REPAIRED", "UNABLE"}:
            raise ValueError(f"unsupported repair status: {self.repair_status}")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must not be negative")
        indexes = [replacement.sentence_index for replacement in self.replacements]
        if len(indexes) != len(set(indexes)):
            raise ValueError("replacement sentence indexes must be unique")
        if len(indexes) > 2:
            raise ValueError("at most two sentences may be replaced")
        if self.repair_status == "REPAIRED" and not self.replacements:
            raise ValueError("REPAIRED result must contain a replacement")
        if self.repair_status == "UNABLE" and self.replacements:
            raise ValueError("UNABLE result must not contain replacements")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceCandidateId": self.source_candidate_id,
            "repairStatus": self.repair_status,
            "replacements": [replacement.to_dict() for replacement in self.replacements],
            "rawOutput": self.raw_output,
            "elapsedMs": round(self.elapsed_ms, 3),
            "usage": self.usage,
            "model": self.model,
            "systemPrompt": self.system_prompt,
            "userPrompt": self.user_prompt,
        }


@runtime_checkable
class PageCandidateRepairer(Protocol):
    async def repair(
        self,
        context: PageGenerationContext,
        profile: Any,
        source_candidate: PageCandidate,
        *,
        repair_plan: dict[str, Any],
    ) -> RepairBatch: ...


class MockPageCandidateGenerator:
    async def repair(
        self,
        context: PageGenerationContext,
        profile: Any,
        source_candidate: PageCandidate,
        *,
        repair_plan: dict[str, Any],
    ) -> RepairBatch:
        editable_indexes = _editable_indexes(repair_plan)
        started = time.perf_counter()
        system_prompt = load_repair_prompt()
        user_prompt = build_repair_user_prompt(
            context=context,
            profile=profile,
            source_candidate=source_candidate,
            repair_plan=repair_plan,
        )
        if not editable_indexes:
            document = {
                "source_candidate_id": source_candidate.candidate_id,
                "repair_status": "UNABLE",
                "replacements": [],
            }
            replacements: tuple[RepairReplacement, ...] = ()
            repair_status: RepairStatus = "UNABLE"
        else:
            subject = context.characters[0] if context.characters else "친구"
            spoken = context.child_input.strip().strip("“”\"'‘’")
            spoken = spoken or "우리 함께 천천히 가요!"
            if spoken[-1] not in ".!?":
                spoken += "!"
            replacement = RepairReplacement(
                sentence_index=editable_indexes[0],
                sentence=f"{subject}는 “{spoken}”라고 다정하게 말해요.",
            )
            replacements = (replacement,)
            repair_status = "REPAIRED"
            document = {
                "source_candidate_id": source_candidate.candidate_id,
                "repair_status": repair_status,
                "replacements": [
                    {
                        "sentence_index": replacement.sentence_index,
                        "sentence": replacement.sentence,
                    }
                ],
            }
        return RepairBatch(
            source_candidate_id=source_candidate.candidate_id,
            repair_status=repair_status,
            replacements=replacements,
            raw_output=json.dumps(document, ensure_ascii=False),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model="mock",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


class OpenAIPageCandidateGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 28.0,
        max_output_tokens: int = 1800,
        client: httpx.AsyncClient | None = None,
        repair_prompt_path: Path | None = None,
    ) -> None:
        _require_nonblank(api_key, "api_key")
        _require_nonblank(model, "model")
        _require_nonblank(base_url, "base_url")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens < 256:
            raise ValueError("max_output_tokens must be at least 256")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._client = client
        self._repair_prompt_path = repair_prompt_path

    async def repair(
        self,
        context: PageGenerationContext,
        profile: Any,
        source_candidate: PageCandidate,
        *,
        repair_plan: dict[str, Any],
    ) -> RepairBatch:
        editable_indexes = _editable_indexes(repair_plan)
        if not editable_indexes:
            raise ValueError("repair plan must contain an editable sentence index")
        system_prompt = load_repair_prompt(self._repair_prompt_path)
        user_prompt = build_repair_user_prompt(
            context=context,
            profile=profile,
            source_candidate=source_candidate,
            repair_plan=repair_plan,
        )
        payload = {
            "model": self._model,
            "store": False,
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
                    "name": "iread_story_page_repair",
                    "strict": True,
                    "schema": _repair_schema(
                        source_candidate.candidate_id,
                        editable_indexes,
                    ),
                }
            },
            "max_output_tokens": min(self._max_output_tokens, 900),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        response = await self._post(payload=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            raise PageGenerationError(
                (f"OpenAI Responses API repair request failed with status {response.status_code}"),
                retryable=(response.status_code in {408, 409, 429} or response.status_code >= 500),
            )

        try:
            data = response.json()
            if data.get("status") not in {None, "completed"}:
                raise ValueError("repair response did not complete")
            raw_output = _extract_output_text(data)
            document = json.loads(raw_output)
            repair_status, replacements = _parse_repair(
                document,
                source_candidate_id=source_candidate.candidate_id,
                editable_indexes=editable_indexes,
            )
            usage = data.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PageGenerationError(
                (
                    "OpenAI returned an invalid story-page repair document "
                    f"({type(exc).__name__}: {exc})"
                ),
                retryable=False,
                raw_output=locals().get("raw_output"),
            ) from exc

        return RepairBatch(
            source_candidate_id=source_candidate.candidate_id,
            repair_status=repair_status,
            replacements=replacements,
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
            raise PageGenerationError(
                "story-page repair timed out",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise PageGenerationError(
                "story-page repair model is unavailable",
                retryable=True,
            ) from exc


class PageGenerationError(RuntimeError):
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


def _repair_schema(
    source_candidate_id: str,
    editable_indexes: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_candidate_id",
            "repair_status",
            "replacements",
        ],
        "properties": {
            "source_candidate_id": {
                "type": "string",
                "enum": [source_candidate_id],
            },
            "repair_status": {
                "type": "string",
                "enum": ["REPAIRED", "UNABLE"],
            },
            "replacements": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(2, len(editable_indexes)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sentence_index", "sentence"],
                    "properties": {
                        "sentence_index": {
                            "type": "integer",
                            "enum": list(editable_indexes),
                        },
                        "sentence": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                },
            },
        },
    }


def _editable_indexes(
    repair_plan: dict[str, Any],
) -> tuple[int, ...]:
    raw_indexes = repair_plan.get("editable_sentence_indexes", [])
    if not isinstance(raw_indexes, list | tuple):
        raise TypeError("editable_sentence_indexes must be an array")
    indexes = tuple(int(index) for index in raw_indexes)
    if len(indexes) > 2:
        raise ValueError("at most two sentence indexes may be editable")
    if len(indexes) != len(set(indexes)):
        raise ValueError("editable sentence indexes must be unique")
    if any(index not in {1, 2, 3, 4} for index in indexes):
        raise ValueError("editable sentence indexes must be between 1 and 4")
    return indexes


def _parse_repair(
    document: dict[str, Any],
    *,
    source_candidate_id: str,
    editable_indexes: tuple[int, ...],
) -> tuple[RepairStatus, tuple[RepairReplacement, ...]]:
    if str(document["source_candidate_id"]).strip() != source_candidate_id:
        raise ValueError("repair source candidate does not match")
    raw_status = str(document["repair_status"]).strip().upper()
    if raw_status not in {"REPAIRED", "UNABLE"}:
        raise ValueError("unsupported repair status")
    raw_replacements = document["replacements"]
    if not isinstance(raw_replacements, list):
        raise TypeError("repair replacements must be an array")
    replacements = tuple(
        RepairReplacement(
            sentence_index=int(item["sentence_index"]),
            sentence=str(item["sentence"]).strip(),
        )
        for item in raw_replacements
        if isinstance(item, dict)
    )
    if len(replacements) != len(raw_replacements):
        raise TypeError("each repair replacement must be an object")
    if any(replacement.sentence_index not in editable_indexes for replacement in replacements):
        raise ValueError("repair changed a locked sentence")
    return cast(RepairStatus, raw_status), replacements


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
                raise ValueError("model refused the story-page repair request")
    output_text = "".join(chunks)
    if not output_text:
        raise ValueError("response contained no output text")
    return output_text


__all__ = [
    "MockPageCandidateGenerator",
    "OpenAIPageCandidateGenerator",
    "PageCandidate",
    "PageCandidateRepairer",
    "PageGenerationContext",
    "PageGenerationError",
    "RepairBatch",
    "RepairReplacement",
    "RepairStatus",
]
