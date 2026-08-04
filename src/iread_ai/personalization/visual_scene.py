from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import httpx

from iread_ai.contracts.story_chapter import (
    StoryChapterEventPayload,
    StoryChapterGenerateRequest,
    StoryVisualScenePayload,
)
from iread_ai.contracts.story_page import StoryCharacterPayload
from iread_ai.personalization.page_splitter import DynamicStoryPage

DEFAULT_VISUAL_SCENE_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "visual_scene.md"
)
_EMOTION_TYPES = (
    "CALM",
    "HAPPY",
    "EXCITED",
    "CURIOUS",
    "SURPRISED",
    "WORRIED",
    "AFRAID",
    "SAD",
    "DISAPPOINTED",
    "RELIEVED",
    "CONFIDENT",
    "FOCUSED",
    "ANGRY",
)
_EMOTION_INTENSITIES = ("LOW", "MEDIUM", "HIGH")
_SHOT_TYPES = (
    "WIDE_ESTABLISHING",
    "WIDE_THREE_QUARTER",
    "MEDIUM_TWO_SHOT",
    "MEDIUM_FULL",
    "CLOSE_UP",
)
_GROUP_MARKERS = ("둘", "두 친구", "함께", "나란히", "서로")
_MOVEMENT_MARKERS = (
    "가요",
    "가요.",
    "걸어",
    "달려",
    "올라",
    "내려",
    "움직",
    "다가가",
    "지나가",
    "따라가",
)
_LOW_INTENSITY_MARKERS = (
    "조용",
    "살짝",
    "천천",
    "낮게",
    "조금",
    "잠깐",
)
_HIGH_INTENSITY_MARKERS = (
    "매우",
    "몹시",
    "벌벌",
    "엉엉",
    "펄쩍",
)
_EMOTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AFRAID", ("무서", "두려", "겁", "떨")),
    ("WORRIED", ("걱정", "불안", "망설", "조마조마")),
    ("SAD", ("슬퍼", "슬프", "울먹", "눈물")),
    ("DISAPPOINTED", ("실망", "풀이 죽", "아쉬워")),
    ("ANGRY", ("화가", "화나", "성내", "짜증")),
    ("SURPRISED", ("놀라", "깜짝", "눈이 커", "입을 벌")),
    ("RELIEVED", ("안도", "다행", "마음을 놓")),
    ("EXCITED", ("신나", "들떠", "가슴이 뛰", "환호")),
    ("HAPPY", ("기뻐", "기쁘", "웃", "미소")),
    ("CURIOUS", ("궁금", "살펴", "고개를 갸웃", "호기심")),
    (
        "CONFIDENT",
        (
            "자신",
            "당당",
            "코웃음",
            "이길 거",
            "자랑",
            "얕보",
            "우쭐",
            "뽐내",
            "도발",
            "훨씬 빨",
            "코를 세워",
        ),
    ),
    (
        "FOCUSED",
        (
            "집중",
            "조심",
            "숨을 고르",
            "기다",
            "준비",
            "끝까지",
            "천천히",
            "발을 옮",
            "출발선",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class VisualSceneGenerationBatch:
    scenes: tuple[dict[str, Any], ...]
    raw_output: str
    elapsed_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    api_call_count: int = 0

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ValueError("visual scene batch must contain at least one scene")
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must not be negative")
        if self.api_call_count < 0:
            raise ValueError("api_call_count must not be negative")


@runtime_checkable
class VisualScenePlanner(Protocol):
    async def generate(
        self,
        *,
        request: StoryChapterGenerateRequest,
        pages: Sequence[DynamicStoryPage],
        question: str | None,
        choices: Sequence[str],
    ) -> VisualSceneGenerationBatch: ...


class MockVisualScenePlanner:
    async def generate(
        self,
        *,
        request: StoryChapterGenerateRequest,
        pages: Sequence[DynamicStoryPage],
        question: str | None,
        choices: Sequence[str],
    ) -> VisualSceneGenerationBatch:
        del question, choices
        started = time.perf_counter()
        scenes = build_chapter_visual_scenes(
            request=request,
            pages=pages,
        )
        return VisualSceneGenerationBatch(
            scenes=scenes,
            raw_output=json.dumps(
                {"pages": list(scenes)},
                ensure_ascii=False,
            ),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            model="mock",
            system_prompt=load_visual_scene_prompt(),
            user_prompt=build_visual_scene_user_prompt(
                request=request,
                pages=pages,
                question=None,
                choices=(),
            ),
            api_call_count=0,
        )


class OpenAIVisualScenePlanner:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 28.0,
        max_output_tokens: int = 3200,
        client: httpx.AsyncClient | None = None,
        prompt_path: Path | None = None,
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

    async def generate(
        self,
        *,
        request: StoryChapterGenerateRequest,
        pages: Sequence[DynamicStoryPage],
        question: str | None,
        choices: Sequence[str],
    ) -> VisualSceneGenerationBatch:
        page_tuple = tuple(pages)
        if not page_tuple:
            raise ValueError("pages must not be empty")
        system_prompt = load_visual_scene_prompt(self._prompt_path)
        user_prompt = build_visual_scene_user_prompt(
            request=request,
            pages=page_tuple,
            question=question,
            choices=choices,
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
                    "name": "iread_story_visual_scenes",
                    "strict": True,
                    "schema": _visual_scene_schema(
                        page_count=len(page_tuple),
                        character_ids=tuple(
                            character.character_id for character in request.story_state.characters
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
        response = await self._post(payload=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            raise VisualSceneGenerationError(
                (
                    "OpenAI Responses API visual-scene request failed "
                    f"with status {response.status_code}"
                ),
                retryable=(response.status_code in {408, 409, 429} or response.status_code >= 500),
            )

        try:
            data = response.json()
            if data.get("status") not in {None, "completed"}:
                raise ValueError("response did not complete")
            raw_output = _extract_output_text(data)
            document = json.loads(raw_output)
            scenes = _parse_visual_scenes(
                document,
                pages=page_tuple,
                character_ids=tuple(
                    character.character_id for character in request.story_state.characters
                ),
            )
            usage = data.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VisualSceneGenerationError(
                (f"OpenAI returned an invalid visual-scene document ({type(exc).__name__}: {exc})"),
                retryable=False,
                raw_output=locals().get("raw_output"),
            ) from exc

        return VisualSceneGenerationBatch(
            scenes=scenes,
            raw_output=raw_output,
            elapsed_ms=elapsed_ms,
            usage=cast(dict[str, Any], usage),
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_call_count=1,
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
            raise VisualSceneGenerationError(
                "visual-scene generation timed out",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise VisualSceneGenerationError(
                "visual-scene model is unavailable",
                retryable=True,
            ) from exc


class VisualSceneGenerationError(RuntimeError):
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


def load_visual_scene_prompt(path: Path | None = None) -> str:
    prompt_path = path or DEFAULT_VISUAL_SCENE_PROMPT_PATH
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("visual-scene prompt must not be empty")
    if len(prompt.encode("utf-8")) > 64 * 1024:
        raise ValueError("visual-scene prompt must not exceed 64 KiB")
    return prompt


def build_visual_scene_user_prompt(
    *,
    request: StoryChapterGenerateRequest,
    pages: Sequence[DynamicStoryPage],
    question: str | None,
    choices: Sequence[str],
) -> str:
    document = {
        "story": {
            "title": request.story_template.title,
            "context": request.story_template.context,
            "chapter_number": request.chapter_number,
            "chapter_goal": request.story_template.current_beat.goal,
            "rolling_summary": request.story_state.rolling_summary,
            "recent_pages": [
                page.model_dump(mode="json", by_alias=True)
                for page in request.story_state.recent_pages
            ],
        },
        "character_catalog": [
            character.model_dump(mode="json", by_alias=True)
            for character in request.story_state.characters
        ],
        "final_pages": [
            {
                "pageNumber": page.page_number,
                "sentences": list(page.sentences),
                "question": (question if page.page_number == len(pages) else None),
                "choices": (list(choices) if page.page_number == len(pages) else []),
            }
            for page in pages
        ],
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _visual_scene_schema(
    *,
    page_count: int,
    character_ids: Sequence[str],
) -> dict[str, Any]:
    character_id_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
    }
    if character_ids:
        character_id_schema["enum"] = list(character_ids)
    character_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "characterId",
            "present",
            "position",
            "orientation",
            "gazeTarget",
            "action",
            "emotion",
        ],
        "properties": {
            "characterId": character_id_schema,
            "present": {"type": "boolean"},
            "position": {"type": ["string", "null"]},
            "orientation": {"type": ["string", "null"]},
            "gazeTarget": {"type": ["string", "null"]},
            "action": {"type": ["string", "null"]},
            "emotion": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "intensity"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": list(_EMOTION_TYPES),
                            },
                            "intensity": {
                                "type": "string",
                                "enum": list(_EMOTION_INTENSITIES),
                            },
                        },
                    },
                    {"type": "null"},
                ]
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["pages"],
        "properties": {
            "pages": {
                "type": "array",
                "minItems": page_count,
                "maxItems": page_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pageNumber", "visualScene"],
                    "properties": {
                        "pageNumber": {
                            "type": "integer",
                            "enum": list(range(1, page_count + 1)),
                        },
                        "visualScene": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "shot",
                                "characters",
                                "mustInclude",
                                "mustAvoid",
                            ],
                            "properties": {
                                "shot": {
                                    "type": "string",
                                    "enum": list(_SHOT_TYPES),
                                },
                                "characters": {
                                    "type": "array",
                                    "minItems": len(character_ids),
                                    "maxItems": len(character_ids),
                                    "items": character_schema,
                                },
                                "mustInclude": {
                                    "type": "array",
                                    "maxItems": 20,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                                "mustAvoid": {
                                    "type": "array",
                                    "maxItems": 20,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    }


def _parse_visual_scenes(
    document: Mapping[str, Any],
    *,
    pages: Sequence[DynamicStoryPage],
    character_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    raw_pages = document["pages"]
    if not isinstance(raw_pages, list) or len(raw_pages) != len(pages):
        raise ValueError("visual-scene page count does not match final pages")
    expected_page_numbers = [page.page_number for page in pages]
    parsed_scenes: list[dict[str, Any]] = []
    for index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, Mapping):
            raise TypeError("visual-scene page must be an object")
        if raw_page["pageNumber"] != expected_page_numbers[index]:
            raise ValueError("visual-scene pages must retain final page order")
        raw_scene = raw_page["visualScene"]
        scene = StoryVisualScenePayload.model_validate(raw_scene)
        output_ids = [character.character_id for character in scene.characters]
        if output_ids != list(character_ids):
            raise ValueError("visual-scene characters must match the catalog order")
        parsed_scenes.append(scene.model_dump(mode="json", by_alias=True))
    return tuple(parsed_scenes)


def _extract_output_text(data: Mapping[str, Any]) -> str:
    convenience_text = data.get("output_text")
    if isinstance(convenience_text, str) and convenience_text:
        return convenience_text
    chunks: list[str] = []
    output = data.get("output", [])
    if not isinstance(output, list):
        raise ValueError("response output must be an array")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content_items = item.get("content", [])
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif content.get("type") == "refusal":
                raise ValueError("model refused the visual-scene request")
    output_text = "".join(chunks)
    if not output_text:
        raise ValueError("response contained no output text")
    return output_text


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def build_chapter_visual_scenes(
    *,
    request: StoryChapterGenerateRequest,
    pages: Sequence[DynamicStoryPage],
) -> tuple[dict[str, Any], ...]:
    characters = tuple(request.story_state.characters)
    events = tuple(request.chapter_plan.ordered_events)
    previous_present: set[str] = set()
    scenes: list[dict[str, Any]] = []

    for page in pages:
        event = _event_for_page(
            events,
            page_number=page.page_number,
            page_count=len(pages),
        )
        page_text = " ".join(page.sentences)
        present_ids = {
            character.character_id for character in characters if character.name in page_text
        }
        present_ids.update(event.required_characters)
        if any(marker in page_text for marker in _GROUP_MARKERS):
            present_ids.update(
                event.required_characters
                or tuple(character.character_id for character in characters)
            )
        if not present_ids and previous_present:
            present_ids.update(previous_present)
        if not present_ids and len(characters) == 1:
            present_ids.add(characters[0].character_id)

        visible = tuple(
            character for character in characters if character.character_id in present_ids
        )
        positions = _positions(visible)
        visual_characters: list[dict[str, Any]] = []
        must_include: list[str] = []

        for character in characters:
            if character.character_id not in present_ids:
                visual_characters.append(
                    {
                        "characterId": character.character_id,
                        "present": False,
                        "position": None,
                        "orientation": None,
                        "gazeTarget": None,
                        "action": None,
                        "emotion": None,
                    }
                )
                continue

            action = _action_for_character(
                character,
                page.sentences,
                event,
            )
            emotion_type = _emotion_type(
                character,
                page.sentences,
                fallback=page_text,
            )
            gaze_target = _gaze_target(
                character,
                visible,
                action,
                event,
            )
            visual_characters.append(
                {
                    "characterId": character.character_id,
                    "present": True,
                    "position": positions[character.character_id],
                    "orientation": (
                        "이동 방향"
                        if any(marker in action for marker in _MOVEMENT_MARKERS)
                        else "시선 대상 방향"
                    ),
                    "gazeTarget": gaze_target,
                    "action": action,
                    "emotion": {
                        "type": emotion_type,
                        "intensity": _emotion_intensity(
                            emotion_type,
                            action,
                        ),
                    },
                }
            )
            must_include.append(f"{character.name}: {action}")

        must_include.extend(event.required_concepts)
        must_avoid = [
            f"{character.name} 등장"
            for character in characters
            if character.character_id not in present_ids
        ]
        must_avoid.extend(
            (
                "본문에 없는 새 인물",
                "글자와 말풍선",
                "로고와 워터마크",
                "같은 캐릭터 중복",
                "신체 왜곡",
            )
        )
        scenes.append(
            {
                "shot": _shot_for_count(len(visible)),
                "characters": visual_characters,
                "mustInclude": _unique(must_include)[:20],
                "mustAvoid": _unique(must_avoid)[:20],
            }
        )
        previous_present = present_ids

    return tuple(scenes)


def _event_for_page(
    events: Sequence[StoryChapterEventPayload],
    *,
    page_number: int,
    page_count: int,
) -> StoryChapterEventPayload:
    index = min(
        len(events) - 1,
        (page_number - 1) * len(events) // max(1, page_count),
    )
    return events[index]


def _positions(
    characters: Sequence[StoryCharacterPayload],
) -> dict[str, str]:
    if len(characters) == 1:
        labels = ("화면 중앙",)
    elif len(characters) == 2:
        labels = ("화면 왼쪽", "화면 오른쪽")
    elif len(characters) == 3:
        labels = ("화면 왼쪽", "화면 중앙", "화면 오른쪽")
    else:
        labels = tuple(
            f"화면 {index + 1}/{len(characters)} 지점" for index in range(len(characters))
        )
    return {character.character_id: labels[index] for index, character in enumerate(characters)}


def _action_for_character(
    character: StoryCharacterPayload,
    sentences: Sequence[str],
    event: StoryChapterEventPayload,
) -> str:
    named = [sentence.strip() for sentence in sentences if character.name in sentence]
    if named:
        return min(
            named,
            key=lambda sentence: (
                int(not _is_character_subject(character.name, sentence)),
                int(any(mark in sentence for mark in ('"', "“", "”"))),
                len(sentence),
            ),
        )
    grouped = [
        sentence.strip()
        for sentence in sentences
        if any(marker in sentence for marker in _GROUP_MARKERS)
    ]
    if grouped:
        return grouped[0]
    return event.locked_event.strip()


def _is_character_subject(name: str, sentence: str) -> bool:
    normalized = sentence.lstrip(" \"'“‘")
    return any(
        normalized.startswith(f"{name}{particle}") for particle in ("은", "는", "이", "가", "도")
    )


def _emotion_type(
    character: StoryCharacterPayload,
    sentences: Sequence[str],
    *,
    fallback: str,
) -> str:
    related = " ".join(sentence for sentence in sentences if character.name in sentence)
    source = related or fallback
    for emotion_type, keywords in _EMOTION_KEYWORDS:
        if any(keyword in source for keyword in keywords):
            return emotion_type
    return "CALM"


def _emotion_intensity(emotion_type: str, source: str) -> str:
    if any(marker in source for marker in _HIGH_INTENSITY_MARKERS):
        return "HIGH"
    if any(marker in source for marker in _LOW_INTENSITY_MARKERS) or emotion_type in {
        "CALM",
        "FOCUSED",
        "WORRIED",
        "AFRAID",
    }:
        return "LOW"
    return "MEDIUM"


def _gaze_target(
    character: StoryCharacterPayload,
    visible: Sequence[StoryCharacterPayload],
    action: str,
    event: StoryChapterEventPayload,
) -> str:
    for other in visible:
        if other.character_id != character.character_id and other.name in action:
            return other.character_id
    if event.required_concepts:
        return event.required_concepts[0]
    return "현재 행동 대상"


def _shot_for_count(visible_count: int) -> str:
    if visible_count == 1:
        return "MEDIUM_FULL"
    if visible_count == 2:
        return "WIDE_THREE_QUARTER"
    return "WIDE_ESTABLISHING"


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


__all__ = [
    "MockVisualScenePlanner",
    "OpenAIVisualScenePlanner",
    "VisualSceneGenerationBatch",
    "VisualSceneGenerationError",
    "VisualScenePlanner",
    "build_chapter_visual_scenes",
    "build_visual_scene_user_prompt",
    "load_visual_scene_prompt",
]
