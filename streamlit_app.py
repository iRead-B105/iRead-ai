from __future__ import annotations

import base64
import binascii
import json
import os
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx
import streamlit as st

from iread_ai.application.chapter_comparison_builder import (
    build_displayed_chapter_comparison_payload,
)
from iread_ai.config import Settings
from iread_ai.contracts.story_image import (
    StoryImageGenerateRequest,
    StoryImageGenerateResponse,
)
from iread_ai.devtools.dynamic_story_simulator import (
    ChapterDisplayCursor,
    ChapterGenerationCursor,
    DynamicStoryStateError,
    advance_display_cursor,
    apply_chapter_response,
    build_chapter_request,
    displayed_page,
    generation_cursor_after_response,
    initial_display_cursor,
    initial_dynamic_runtime,
    initial_generation_cursor,
)
from iread_ai.devtools.reading_profiles import (
    READING_PROFILE_PRESETS,
)
from iread_ai.devtools.service_story_catalog import (
    STORY_CATALOG,
    ServiceStoryFixture,
    get_story_fixture,
)

DEFAULT_ENDPOINT = os.getenv(
    "IREAD_STORY_CHAPTER_ENDPOINT",
    "http://127.0.0.1:8081/api/v3/story/chapters/generate",
)
DEFAULT_IMAGE_ENDPOINT = os.getenv(
    "IREAD_STORY_IMAGE_ENDPOINT",
    "http://127.0.0.1:8081/api/v1/story/images/generate",
)
PROFILE_KEYS = ("fluent", "balanced", "beginner")
MAX_IMAGE_RESPONSE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_JSON_RESPONSE_BYTES = 17 * 1024 * 1024
MAX_CACHED_PAGE_IMAGES = 4
MAX_PARALLEL_IMAGE_JOBS = 2
IMAGE_POLL_INTERVAL_SECONDS = 0.5


class StoryChapterAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable


def _default_internal_key() -> str:
    try:
        return Settings().internal_api_key.get_secret_value()
    except (OSError, ValueError):
        return os.getenv("INTERNAL_API_KEY", "local-development-key")


@st.cache_resource
def _http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(50.0, connect=5.0))


@st.cache_resource
def _image_http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(200.0, connect=5.0))


@st.cache_resource
def _image_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_IMAGE_JOBS,
        thread_name_prefix="iread-story-image",
    )


def _cancel_pending_image_jobs() -> None:
    jobs = st.session_state.get("dynamic_image_jobs", {})
    if isinstance(jobs, Mapping):
        for raw_job in jobs.values():
            job = raw_job if isinstance(raw_job, Mapping) else {}
            future = job.get("future")
            if isinstance(future, Future):
                future.cancel()
    st.session_state.dynamic_image_jobs = {}


def _cancel_queued_image_jobs() -> None:
    jobs = dict(st.session_state.get("dynamic_image_jobs", {}))
    for image_key, raw_job in list(jobs.items()):
        job = raw_job if isinstance(raw_job, Mapping) else {}
        future = job.get("future")
        if not isinstance(future, Future) or future.cancel():
            jobs.pop(image_key, None)
    st.session_state.dynamic_image_jobs = jobs


def _on_auto_image_change() -> None:
    if not bool(st.session_state.get("dynamic-auto-image", True)):
        _cancel_queued_image_jobs()


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "dynamic_story_id": None,
        "dynamic_profile_key": "balanced",
        "dynamic_runtime": None,
        "dynamic_generation_cursor": initial_generation_cursor(),
        "dynamic_display_cursor": None,
        "dynamic_current_response": None,
        "dynamic_chapter_history": [],
        "dynamic_answer_history": [],
        "dynamic_run_id": uuid.uuid4().hex[:10],
        "dynamic_pending_request": None,
        "dynamic_pending_key": None,
        "dynamic_pending_branch": None,
        "dynamic_last_error": None,
        "dynamic_retryable": False,
        "dynamic_page_images": {},
        "dynamic_image_errors": {},
        "dynamic_image_jobs": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("dynamic-endpoint", DEFAULT_ENDPOINT)
    st.session_state.setdefault(
        "dynamic-image-endpoint",
        DEFAULT_IMAGE_ENDPOINT,
    )
    st.session_state.setdefault("dynamic-api-key", _default_internal_key())
    st.session_state.setdefault("dynamic-auto-image", True)
    st.session_state.setdefault(
        "dynamic-story-selector",
        STORY_CATALOG[0].template_id,
    )
    st.session_state.setdefault("dynamic-profile-selector", "balanced")


def _reset_story() -> None:
    _cancel_pending_image_jobs()
    for key, value in {
        "dynamic_story_id": None,
        "dynamic_runtime": None,
        "dynamic_generation_cursor": initial_generation_cursor(),
        "dynamic_display_cursor": None,
        "dynamic_current_response": None,
        "dynamic_chapter_history": [],
        "dynamic_answer_history": [],
        "dynamic_run_id": uuid.uuid4().hex[:10],
        "dynamic_pending_request": None,
        "dynamic_pending_key": None,
        "dynamic_pending_branch": None,
        "dynamic_last_error": None,
        "dynamic_retryable": False,
        "dynamic_page_images": {},
        "dynamic_image_errors": {},
        "dynamic_image_jobs": {},
    }.items():
        st.session_state[key] = value


def _start_story(
    story: ServiceStoryFixture,
    profile_key: str,
) -> None:
    _cancel_pending_image_jobs()
    st.session_state.dynamic_story_id = story.template_id
    st.session_state.dynamic_profile_key = profile_key
    st.session_state.dynamic_runtime = initial_dynamic_runtime(
        story,
        profile_key=profile_key,
    )
    st.session_state.dynamic_generation_cursor = (
        initial_generation_cursor()
    )
    st.session_state.dynamic_display_cursor = None
    st.session_state.dynamic_current_response = None
    st.session_state.dynamic_chapter_history = []
    st.session_state.dynamic_answer_history = []
    st.session_state.dynamic_pending_request = None
    st.session_state.dynamic_pending_key = None
    st.session_state.dynamic_pending_branch = None
    st.session_state.dynamic_last_error = None
    st.session_state.dynamic_retryable = False
    st.session_state.dynamic_page_images = {}
    st.session_state.dynamic_image_errors = {}
    st.session_state.dynamic_image_jobs = {}


def _post_chapter(
    endpoint: str,
    api_key: str,
    idempotency_key: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], float, bool]:
    if not endpoint.startswith(("http://", "https://")):
        raise StoryChapterAPIError(
            "API 주소는 http:// 또는 https://로 시작해야 합니다."
        )
    if not api_key.strip():
        raise StoryChapterAPIError("X-API-Key를 입력해 주세요.")
    started = time.perf_counter()
    try:
        response = _http_client().post(
            endpoint,
            headers={
                "X-API-Key": api_key.strip(),
                "Idempotency-Key": idempotency_key,
                "Content-Type": "application/json",
            },
            json=dict(payload),
        )
    except httpx.TimeoutException as exc:
        raise StoryChapterAPIError(
            "장 생성 응답이 45초를 넘었습니다.",
            retryable=True,
        ) from exc
    except httpx.RequestError as exc:
        raise StoryChapterAPIError(
            "AI 서버에 연결하지 못했습니다.",
            retryable=True,
        ) from exc
    wall_ms = (time.perf_counter() - started) * 1000
    if response.is_error:
        try:
            error = response.json()
        except ValueError:
            error = {}
        message = str(
            error.get("message")
            or error.get("detail")
            or f"HTTP {response.status_code}"
        )
        code = str(error.get("code") or "")
        raise StoryChapterAPIError(
            f"{code}: {message}" if code else message,
            retryable=(
                bool(error.get("retryable"))
                or response.status_code in {502, 503, 504}
            ),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise StoryChapterAPIError(
            "AI 서버가 JSON이 아닌 응답을 보냈습니다."
        ) from exc
    if not isinstance(body, Mapping):
        raise StoryChapterAPIError("장 생성 응답 형식이 올바르지 않습니다.")
    return (
        {str(key): value for key, value in body.items()},
        wall_ms,
        response.headers.get("Idempotent-Replayed", "").lower() == "true",
    )


def _page_image_key(
    response: Mapping[str, Any],
    page_number: int,
) -> str:
    generation_id = str(response.get("generationId") or "unknown")
    return f"{generation_id}:p{page_number}"


def _image_story_characters(
    chapter_request: Mapping[str, Any],
    chapter_response: Mapping[str, Any],
    visual_scene: Mapping[str, Any],
) -> list[dict[str, Any]]:
    story_state = _as_mapping(chapter_request.get("storyState"))
    merged: dict[str, dict[str, Any]] = {}
    for raw_character in story_state.get("characters", []):
        character = _as_mapping(raw_character)
        character_id = str(character.get("characterId") or "").strip()
        if character_id:
            merged[character_id] = character

    state_patch = _as_mapping(chapter_response.get("statePatch"))
    for raw_character in state_patch.get("charactersUpserted", []):
        character = _as_mapping(raw_character)
        character_id = str(character.get("characterId") or "").strip()
        if character_id:
            merged[character_id] = character

    visual_ids: list[str] = []
    for raw_character in visual_scene.get("characters", []):
        character = _as_mapping(raw_character)
        character_id = str(character.get("characterId") or "").strip()
        if character_id and character_id not in visual_ids:
            visual_ids.append(character_id)
            merged.setdefault(
                character_id,
                {
                    "characterId": character_id,
                    "name": None,
                    "role": "이야기 등장인물",
                    "immutableTraits": [],
                },
            )

    ordered_ids = [
        *visual_ids,
        *(character_id for character_id in merged if character_id not in visual_ids),
    ]
    return [
        _image_character_document(character_id, merged[character_id])
        for character_id in ordered_ids
    ]


def _image_character_document(
    character_id: str,
    character: Mapping[str, Any],
) -> dict[str, Any]:
    role = str(character.get("role") or "이야기 등장인물")
    return {
        "characterId": character_id,
        "name": str(character.get("name") or role or character_id),
        "role": role,
        "immutableTraits": [
            str(value)
            for value in character.get("immutableTraits", [])
            if str(value).strip()
        ],
    }


def _build_image_payload(
    current_record: Mapping[str, Any],
    page: Mapping[str, Any],
) -> dict[str, Any]:
    chapter_request = _as_mapping(current_record.get("request"))
    chapter_response = _as_mapping(current_record.get("response"))
    page_number = int(page["pageNumber"])
    visual_scene = _as_mapping(page.get("visualScene"))
    if not visual_scene:
        raise StoryChapterAPIError(
            "현재 페이지에 visualScene이 없어 그림을 만들 수 없습니다."
        )

    story_template = _as_mapping(chapter_request.get("storyTemplate"))
    characters = _image_story_characters(
        chapter_request,
        chapter_response,
        visual_scene,
    )
    generation_id = str(
        chapter_response.get("generationId") or uuid.uuid4().hex
    )
    payload = {
        "requestId": f"image-{generation_id}-p{page_number}",
        "schemaVersion": 1,
        "storyId": int(chapter_response["storyId"]),
        "storyRevision": int(chapter_response["storyRevision"]),
        "chapterNumber": int(chapter_response["chapterNumber"]),
        "pageNumber": page_number,
        "sentences": [str(value) for value in page["sentences"]],
        "visualScene": visual_scene,
        "storyContext": {
            "title": str(story_template.get("title") or "이야기"),
            "characters": characters,
        },
        "characterReferences": [],
    }
    return StoryImageGenerateRequest.model_validate(payload).model_dump(
        by_alias=True
    )


def _post_page_image(
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, float, bool]:
    if not endpoint.startswith(("http://", "https://")):
        raise StoryChapterAPIError(
            "그림 API 주소는 http:// 또는 https://로 시작해야 합니다."
        )
    if not api_key.strip():
        raise StoryChapterAPIError("X-API-Key를 입력해 주세요.")

    request_model = StoryImageGenerateRequest.model_validate(payload)
    idempotency_key = str(payload["requestId"])
    started = time.perf_counter()
    raw_body = bytearray()
    try:
        with client.stream(
            "POST",
            endpoint,
            headers={
                "X-API-Key": api_key.strip(),
                "Idempotency-Key": idempotency_key,
                "Content-Type": "application/json",
            },
            json=dict(payload),
        ) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    declared_length = -1
                if declared_length > MAX_IMAGE_JSON_RESPONSE_BYTES:
                    raise StoryChapterAPIError(
                        "그림 생성 응답이 허용 크기를 넘었습니다."
                    )
            for chunk in response.iter_bytes():
                if (
                    len(raw_body) + len(chunk)
                    > MAX_IMAGE_JSON_RESPONSE_BYTES
                ):
                    raise StoryChapterAPIError(
                        "그림 생성 응답이 허용 크기를 넘었습니다."
                    )
                raw_body.extend(chunk)
            response_status = response.status_code
            replayed = (
                response.headers.get("Idempotent-Replayed", "").lower()
                == "true"
            )
    except httpx.TimeoutException as exc:
        raise StoryChapterAPIError(
            "그림 생성 응답이 200초를 넘었습니다.",
            retryable=True,
        ) from exc
    except httpx.RequestError as exc:
        raise StoryChapterAPIError(
            "그림 생성 API에 연결하지 못했습니다.",
            retryable=True,
        ) from exc
    wall_ms = (time.perf_counter() - started) * 1000

    try:
        body = json.loads(raw_body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise StoryChapterAPIError(
            "그림 생성 API가 JSON이 아닌 응답을 보냈습니다."
        ) from exc
    if not isinstance(body, Mapping):
        raise StoryChapterAPIError("그림 생성 응답 형식이 올바르지 않습니다.")

    if response_status >= 400:
        error = body
        message = str(
            error.get("message")
            or error.get("detail")
            or f"HTTP {response_status}"
        )
        code = str(error.get("code") or "")
        raise StoryChapterAPIError(
            f"{code}: {message}" if code else message,
            retryable=(
                bool(error.get("retryable"))
                or response_status in {429, 502, 503, 504}
            ),
        )

    try:
        response_model = StoryImageGenerateResponse.model_validate(
            body
        ).validate_against_request(request_model)
    except ValueError as exc:
        raise StoryChapterAPIError(
            "그림 생성 응답 계약이 요청과 일치하지 않습니다."
        ) from exc
    validated = response_model.model_dump(by_alias=True)

    mime_type = response_model.mime_type
    encoded = response_model.image_base64
    if len(encoded) > (MAX_IMAGE_RESPONSE_BYTES * 4 // 3) + 8:
        raise StoryChapterAPIError("생성된 그림이 허용 크기를 넘었습니다.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise StoryChapterAPIError(
            "그림 생성 응답의 Base64 형식이 올바르지 않습니다."
        ) from exc
    if not content or len(content) > MAX_IMAGE_RESPONSE_BYTES:
        raise StoryChapterAPIError("생성된 그림 크기가 올바르지 않습니다.")
    if _detected_image_mime(content) != mime_type:
        raise StoryChapterAPIError(
            "그림 MIME 형식과 실제 파일 형식이 일치하지 않습니다."
        )

    safe_body = {
        str(key): value
        for key, value in validated.items()
        if key != "imageBase64"
    }
    return (
        safe_body,
        content,
        wall_ms,
        replayed,
    )


def _detected_image_mime(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if (
        len(content) >= 12
        and content.startswith(b"RIFF")
        and content[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _image_job_error(exc: Exception) -> dict[str, Any]:
    return {
        "message": str(exc),
        "retryable": (
            exc.retryable
            if isinstance(exc, StoryChapterAPIError)
            else False
        ),
    }


def _enqueue_page_image(
    current_record: Mapping[str, Any],
    page: Mapping[str, Any],
) -> bool:
    response = _as_mapping(current_record.get("response"))
    page_number = int(page["pageNumber"])
    image_key = _page_image_key(response, page_number)
    if image_key in st.session_state.dynamic_page_images:
        return False
    existing_job = _as_mapping(
        st.session_state.dynamic_image_jobs.get(image_key)
    )
    existing_future = existing_job.get("future")
    if isinstance(existing_future, Future):
        return False

    try:
        payload = _build_image_payload(current_record, page)
        endpoint = str(
            st.session_state["dynamic-image-endpoint"]
        ).strip()
        api_key = str(st.session_state["dynamic-api-key"])
        client = _image_http_client()
        future = _image_executor().submit(
            _post_page_image,
            client,
            endpoint,
            api_key,
            payload,
        )
        jobs = dict(st.session_state.dynamic_image_jobs)
        jobs[image_key] = {
            "future": future,
            "request": payload,
            "pageNumber": page_number,
            "generationId": str(response.get("generationId") or ""),
            "submittedAt": time.time(),
        }
        st.session_state.dynamic_image_jobs = jobs
        errors = dict(st.session_state.dynamic_image_errors)
        errors.pop(image_key, None)
        st.session_state.dynamic_image_errors = errors
        return True
    except (
        KeyError,
        TypeError,
        ValueError,
        StoryChapterAPIError,
    ) as exc:
        errors = dict(st.session_state.dynamic_image_errors)
        errors[image_key] = _image_job_error(exc)
        st.session_state.dynamic_image_errors = errors
        return False


def _enqueue_missing_chapter_images(
    current_record: Mapping[str, Any],
) -> int:
    response = _as_mapping(current_record.get("response"))
    pages = response.get("pages", [])
    if not isinstance(pages, list):
        return 0
    enqueued = 0
    for raw_page in pages[:MAX_CACHED_PAGE_IMAGES]:
        page = _as_mapping(raw_page)
        if not page:
            continue
        page_number = int(page.get("pageNumber") or 0)
        image_key = _page_image_key(response, page_number)
        if image_key in st.session_state.dynamic_image_errors:
            continue
        if _enqueue_page_image(current_record, page):
            enqueued += 1
    return enqueued


def _start_chapter_image_jobs(
    current_record: Mapping[str, Any],
) -> int:
    _cancel_pending_image_jobs()
    st.session_state.dynamic_page_images = {}
    st.session_state.dynamic_image_errors = {}
    if not bool(st.session_state.get("dynamic-auto-image", True)):
        return 0
    return _enqueue_missing_chapter_images(current_record)


def _harvest_image_jobs() -> int:
    jobs = dict(st.session_state.get("dynamic_image_jobs", {}))
    images = dict(st.session_state.get("dynamic_page_images", {}))
    errors = dict(st.session_state.get("dynamic_image_errors", {}))
    completed = 0
    for image_key, raw_job in list(jobs.items()):
        job = _as_mapping(raw_job)
        future = job.get("future")
        if not isinstance(future, Future) or not future.done():
            continue
        jobs.pop(image_key, None)
        completed += 1
        try:
            metadata, content, wall_ms, replayed = future.result()
            images.pop(image_key, None)
            images[image_key] = {
                "request": _as_mapping(job.get("request")),
                "response": metadata,
                "content": content,
                "wallMs": wall_ms,
                "replayed": replayed,
            }
            errors.pop(image_key, None)
        except Exception as exc:
            errors[image_key] = _image_job_error(exc)

    while len(images) > MAX_CACHED_PAGE_IMAGES:
        oldest_key = next(iter(images))
        images.pop(oldest_key, None)
    st.session_state.dynamic_image_jobs = jobs
    st.session_state.dynamic_page_images = images
    st.session_state.dynamic_image_errors = errors
    return completed


def _chapter_image_progress(
    response: Mapping[str, Any],
) -> dict[str, int]:
    pages = response.get("pages", [])
    page_numbers = [
        int(page["pageNumber"])
        for page in pages
        if isinstance(page, Mapping) and "pageNumber" in page
    ]
    keys = {
        _page_image_key(response, page_number)
        for page_number in page_numbers
    }
    images = st.session_state.get("dynamic_page_images", {})
    errors = st.session_state.get("dynamic_image_errors", {})
    jobs = st.session_state.get("dynamic_image_jobs", {})
    running = 0
    queued = 0
    for image_key in keys:
        job = jobs.get(image_key, {}) if isinstance(jobs, Mapping) else {}
        future = job.get("future") if isinstance(job, Mapping) else None
        if not isinstance(future, Future) or future.done():
            continue
        if future.running():
            running += 1
        else:
            queued += 1
    return {
        "total": len(keys),
        "ready": sum(
            image_key in images
            for image_key in keys
        ),
        "failed": sum(
            image_key in errors
            for image_key in keys
        ),
        "running": running,
        "queued": queued,
    }


def _current_page_image(
    response: Mapping[str, Any],
    page_number: int,
) -> dict[str, Any]:
    return _as_mapping(
        st.session_state.dynamic_page_images.get(
            _page_image_key(response, page_number)
        )
    )


def _current_page_image_error(
    response: Mapping[str, Any],
    page_number: int,
) -> dict[str, Any]:
    return _as_mapping(
        st.session_state.dynamic_image_errors.get(
            _page_image_key(response, page_number)
        )
    )


def _comparison_endpoint(chapter_endpoint: str) -> str:
    base_url = chapter_endpoint.split("/api/", maxsplit=1)[0].rstrip("/")
    return f"{base_url}/api/dev/story/displayed-chapter-comparison"


def _health_endpoint(chapter_endpoint: str) -> str:
    base_url = chapter_endpoint.split("/api/", maxsplit=1)[0].rstrip("/")
    return f"{base_url}/health"


@st.cache_data(ttl=5, max_entries=8)
def _get_server_health(chapter_endpoint: str) -> dict[str, str]:
    endpoint = _health_endpoint(chapter_endpoint)
    try:
        response = httpx.get(endpoint, timeout=3.0)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return {"status": "unavailable", "storyProvider": "unknown"}
    if not isinstance(body, Mapping):
        return {"status": "invalid", "storyProvider": "unknown"}
    return {
        "status": str(body.get("status") or "unknown"),
        "storyProvider": str(body.get("storyProvider") or "unknown"),
    }


def _post_comparison(
    chapter_payload: Mapping[str, Any],
    personalized_response: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    endpoint = _comparison_endpoint(
        str(st.session_state["dynamic-endpoint"]).strip()
    )
    payload = build_displayed_chapter_comparison_payload(
        chapter_payload,
        personalized_response,
        request_id=f"compare-chapter-{uuid.uuid4().hex}",
    )
    started = time.perf_counter()
    try:
        response = _http_client().post(
            endpoint,
            headers={
                "X-API-Key": str(
                    st.session_state["dynamic-api-key"]
                ).strip(),
                "Content-Type": "application/json",
            },
            json=payload,
        )
    except httpx.TimeoutException as exc:
        raise StoryChapterAPIError(
            "기본 LLM 비교 응답이 45초를 넘었습니다.",
            retryable=True,
        ) from exc
    except httpx.RequestError as exc:
        raise StoryChapterAPIError(
            "기본 LLM 비교 API에 연결하지 못했습니다.",
            retryable=True,
        ) from exc
    wall_ms = (time.perf_counter() - started) * 1000
    if response.is_error:
        try:
            error = response.json()
        except ValueError:
            error = {}
        message = str(
            error.get("message")
            or error.get("detail")
            or f"HTTP {response.status_code}"
        )
        code = str(error.get("code") or "")
        raise StoryChapterAPIError(
            f"{code}: {message}" if code else message,
            retryable=(
                bool(error.get("retryable"))
                or response.status_code in {502, 503, 504}
            ),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise StoryChapterAPIError(
            "기본 LLM 비교 API가 JSON이 아닌 응답을 보냈습니다."
        ) from exc
    if not isinstance(body, Mapping):
        raise StoryChapterAPIError(
            "기본 LLM 비교 응답 형식이 올바르지 않습니다."
        )
    return (
        {str(key): value for key, value in body.items()},
        wall_ms,
    )


def _compare_record(record_index: int) -> bool:
    records = [
        dict(record)
        for record in st.session_state.dynamic_chapter_history
    ]
    if record_index < 0 or record_index >= len(records):
        return False
    record = records[record_index]
    try:
        comparison, wall_ms = _post_comparison(
            _as_mapping(record.get("request")),
            _as_mapping(record.get("response")),
        )
        record["comparison"] = comparison
        record["comparisonWallMs"] = wall_ms
        record["comparisonError"] = None
        records[record_index] = record
        st.session_state.dynamic_chapter_history = records
        return True
    except (
        StoryChapterAPIError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        record["comparisonError"] = str(exc)
        records[record_index] = record
        st.session_state.dynamic_chapter_history = records
        return False


def _prepare_request(
    story: ServiceStoryFixture,
    branch_input: Mapping[str, str] | None,
) -> None:
    cursor: ChapterGenerationCursor = (
        st.session_state.dynamic_generation_cursor
    )
    runtime = st.session_state.dynamic_runtime
    request_id = (
        f"dynamic-{st.session_state.dynamic_run_id}-"
        f"r{runtime['storyRevision']}-c{cursor.chapter_index + 1}-"
        f"{uuid.uuid4().hex[:6]}"
    )
    st.session_state.dynamic_pending_request = build_chapter_request(
        story,
        runtime,
        cursor.chapter_index,
        branch_input,
        request_id,
    )
    st.session_state.dynamic_pending_key = request_id
    st.session_state.dynamic_pending_branch = (
        dict(branch_input) if branch_input is not None else None
    )


def _consume_request(story: ServiceStoryFixture) -> None:
    payload = st.session_state.dynamic_pending_request
    key = st.session_state.dynamic_pending_key
    if not isinstance(payload, Mapping) or not key:
        raise StoryChapterAPIError("재시도할 장 생성 요청이 없습니다.")
    response, wall_ms, replayed = _post_chapter(
        str(st.session_state["dynamic-endpoint"]).strip(),
        str(st.session_state["dynamic-api-key"]),
        str(key),
        payload,
    )
    runtime = apply_chapter_response(
        st.session_state.dynamic_runtime,
        response,
    )
    generation_cursor = generation_cursor_after_response(
        story,
        st.session_state.dynamic_generation_cursor,
        response,
    )
    display_cursor = initial_display_cursor(response)
    record = {
        "request": dict(payload),
        "response": response,
        "wallMs": wall_ms,
        "replayed": replayed,
        "comparison": None,
        "comparisonWallMs": None,
        "comparisonError": None,
    }
    st.session_state.dynamic_runtime = runtime
    st.session_state.dynamic_generation_cursor = generation_cursor
    st.session_state.dynamic_display_cursor = display_cursor
    st.session_state.dynamic_current_response = response
    st.session_state.dynamic_chapter_history = [
        *st.session_state.dynamic_chapter_history,
        record,
    ]
    _start_chapter_image_jobs(record)
    branch = st.session_state.dynamic_pending_branch
    if branch is not None:
        previous_response = (
            st.session_state.dynamic_chapter_history[-2]["response"]
            if len(st.session_state.dynamic_chapter_history) >= 2
            else {}
        )
        previous_pages = previous_response.get("pages", [])
        previous_question = (
            previous_pages[-1].get("question")
            if previous_pages
            else None
        )
        st.session_state.dynamic_answer_history = [
            *st.session_state.dynamic_answer_history,
            {
                "question": previous_question,
                "answer": branch["text"],
                "source": branch["source"],
            },
        ]
    st.session_state.dynamic_pending_request = None
    st.session_state.dynamic_pending_key = None
    st.session_state.dynamic_pending_branch = None
    st.session_state.dynamic_last_error = None
    st.session_state.dynamic_retryable = False


def _generate_chapter(
    story: ServiceStoryFixture,
    branch_input: Mapping[str, str] | None,
    *,
    retry: bool = False,
) -> bool:
    try:
        if not retry:
            _prepare_request(story, branch_input)
        _consume_request(story)
        return True
    except (
        DynamicStoryStateError,
        StoryChapterAPIError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        st.session_state.dynamic_last_error = str(exc)
        st.session_state.dynamic_retryable = (
            exc.retryable
            if isinstance(exc, StoryChapterAPIError)
            else False
        )
        return False


def _generate_with_status(
    story: ServiceStoryFixture,
    branch_input: Mapping[str, str] | None,
    *,
    retry: bool = False,
) -> bool:
    with st.status(
        "한 장의 이야기를 만들고 읽기 분량에 맞게 나누고 있어요.",
        expanded=True,
    ) as status:
        st.write("1. 장 전체 문장 후보를 한 번에 생성해요.")
        st.write("2. 후보마다 50~70음절 기준으로 2~4페이지를 나눠요.")
        st.write("3. Kiwi·G2P로 아동 읽기 프로필 적합도를 검사하고 고릅니다.")
        st.write("4. 하드 위반이 남을 때만 한 페이지를 국소 교정해요.")
        success = _generate_chapter(
            story,
            branch_input,
            retry=retry,
        )
        if success:
            response = st.session_state.dynamic_current_response
            page_count = len(response["pages"])
            status.update(
                label=f"{page_count}페이지 장이 준비됐어요.",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label="장 생성에 실패했어요.",
                state="error",
                expanded=True,
            )
        return success


def _render_illustrated_page(
    image_entry: Mapping[str, Any],
) -> bool:
    content = image_entry.get("content")
    metadata = _as_mapping(image_entry.get("response"))
    mime_type = str(metadata.get("mimeType") or "")
    if not isinstance(content, bytes) or mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        return False

    st.image(
        content,
        caption="현재 이야기 장면",
        width="stretch",
    )
    return True


def _render_page_sentences(page: Mapping[str, Any]) -> None:
    with st.container(border=True):
        st.caption("이 페이지에서 읽을 글")
        for sentence in page["sentences"]:
            st.markdown(f"#### {sentence}")


def _render_story_page(
    story: ServiceStoryFixture,
    response: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
    image_entry: Mapping[str, Any],
) -> None:
    page = displayed_page(response, cursor)
    chapter_number = int(response["chapterNumber"])
    total_chapters = story.total_chapters
    progress = (
        chapter_number - 1 + cursor.page_number / cursor.page_count
    ) / total_chapters
    st.caption(
        f"{chapter_number}/{total_chapters}장 · "
        f"{cursor.page_number}/{cursor.page_count}페이지"
    )
    st.progress(min(1.0, progress))
    has_image = _render_illustrated_page(image_entry)
    if has_image:
        st.caption("그림과 읽을 글을 분리해 장면 전체가 보이도록 표시합니다.")
    else:
        st.info(
            "글을 먼저 보여 드려요. 현재 페이지 그림은 이어서 준비합니다.",
            icon=":material/image:",
        )
    _render_page_sentences(page)


def _render_image_generation_control(
    current_record: Mapping[str, Any],
    response: Mapping[str, Any],
    page: Mapping[str, Any],
) -> None:
    page_number = int(page["pageNumber"])
    image_key = _page_image_key(response, page_number)
    progress = _chapter_image_progress(response)
    active_count = progress["running"] + progress["queued"]
    st.caption(
        "장 전체 그림 "
        f"{progress['ready']}/{progress['total']}장 준비 · "
        f"{progress['running']}장 생성 중 · "
        f"{progress['queued']}장 대기 · 최대 "
        f"{MAX_PARALLEL_IMAGE_JOBS}장 병렬"
    )

    image_entry = _current_page_image(response, page_number)
    if image_entry:
        return

    jobs = st.session_state.get("dynamic_image_jobs", {})
    current_job = (
        jobs.get(image_key)
        if isinstance(jobs, Mapping)
        else None
    )
    if isinstance(current_job, Mapping):
        st.info(
            "현재 페이지 그림을 백그라운드에서 만들고 있어요. "
            "글과 페이지 이동은 그대로 사용할 수 있습니다.",
            icon=":material/progress_activity:",
        )
        return

    error = _current_page_image_error(response, page_number)
    if error:
        st.error(
            "글은 정상 생성됐지만 현재 페이지 그림을 만들지 못했습니다. "
            f"{error.get('message', '')}",
            icon=":material/broken_image:",
        )
        if st.button(
            "현재 페이지 그림 다시 만들기",
            icon=":material/refresh:",
            width="stretch",
            key=(
                f"dynamic-retry-image-{response.get('generationId')}-"
                f"{page_number}"
            ),
        ):
            _enqueue_page_image(current_record, page)
            st.rerun()
        return

    auto_image = bool(st.session_state.get("dynamic-auto-image", True))
    if not auto_image:
        if st.button(
            "현재 페이지 그림 만들기",
            icon=":material/image:",
            width="stretch",
            key=(
                f"dynamic-generate-image-{response.get('generationId')}-"
                f"{page_number}"
            ),
        ):
            _enqueue_page_image(current_record, page)
            st.rerun()
        return

    if active_count == 0 and _enqueue_missing_chapter_images(current_record):
        st.rerun()


def _render_story_page_with_images(
    story: ServiceStoryFixture,
    response: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
    current_record: Mapping[str, Any],
) -> None:
    page = displayed_page(response, cursor)
    image_entry = _current_page_image(
        response,
        int(page["pageNumber"]),
    )
    _render_story_page(
        story,
        response,
        cursor,
        image_entry,
    )
    _render_image_generation_control(
        current_record,
        response,
        page,
    )


@st.fragment(run_every=IMAGE_POLL_INTERVAL_SECONDS)
def _render_story_page_polling(
    story: ServiceStoryFixture,
    response: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
    current_record: Mapping[str, Any],
) -> None:
    completed = _harvest_image_jobs()
    _render_story_page_with_images(
        story,
        response,
        cursor,
        current_record,
    )
    if completed:
        st.rerun()


@st.fragment
def _render_story_page_static(
    story: ServiceStoryFixture,
    response: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
    current_record: Mapping[str, Any],
) -> None:
    _harvest_image_jobs()
    _render_story_page_with_images(
        story,
        response,
        cursor,
        current_record,
    )


def _render_answer_form(
    page: Mapping[str, Any],
    *,
    chapter_number: int,
) -> tuple[bool, dict[str, str] | None]:
    question = str(page.get("question") or "")
    choices = [str(choice) for choice in page.get("choices", [])]
    st.subheader(question)
    with st.form(f"dynamic-answer-{chapter_number}"):
        selected = st.segmented_control(
            "개발자 테스트용 선택지",
            choices,
            default=choices[0] if choices else None,
            key=f"dynamic-choice-{chapter_number}",
        )
        custom = st.text_input(
            "실서비스의 STT 입력 대신 내 생각을 적어 주세요",
            key=f"dynamic-custom-{chapter_number}",
        )
        submitted = st.form_submit_button(
            "이 답으로 다음 장 만들기",
            type="primary",
            icon=":material/auto_stories:",
            width="stretch",
        )
    if not submitted:
        return False, None
    text = custom.strip() or str(selected or "").strip()
    if not text:
        st.warning("선택지를 고르거나 내 생각을 적어 주세요.")
        return False, None
    return True, {
        "source": "TEXT_CONFIRMED" if custom.strip() else "CHOICE",
        "text": text,
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _format_ms(value: Any) -> str:
    number = _numeric(value)
    if number is None:
        return "-"
    if number < 1000:
        return f"{number:,.0f}ms"
    return f"{number / 1000:,.2f}초"


def _score_label(value: Any, confidence: Any = "FULL") -> str:
    number = _numeric(value)
    if number is None:
        return "계산 불가"
    suffix = " · 부분" if str(confidence) == "PARTIAL" else ""
    return f"{number:.1f} / 100{suffix}"


def _occurrence_label(value: Any) -> str:
    number = _numeric(value)
    return str(int(number)) if number is not None else "미검증"


def _contract_label(value: Any) -> str:
    if value is True:
        return "통과"
    if value is False:
        return "위반"
    return "미검증"


def _render_chapter_text(pages: Any) -> None:
    if not isinstance(pages, list):
        st.caption("표시할 장 본문이 없습니다.")
        return
    for page in pages:
        document = _as_mapping(page)
        st.markdown(
            f"**{document.get('pageNumber', '-')}페이지**"
        )
        for sentence in document.get("sentences", []):
            st.write(str(sentence))
        question = str(document.get("question") or "").strip()
        if question:
            st.caption(f"질문 · {question}")
        choices = document.get("choices", [])
        if isinstance(choices, list) and choices:
            st.caption(
                "선택지 · "
                + " / ".join(str(choice) for choice in choices)
            )


def _render_chapter_comparison_panel(
    current_record: Mapping[str, Any],
    record_index: int,
) -> None:
    st.subheader("개인화 효과 비교")
    comparison = _as_mapping(current_record.get("comparison"))
    if not comparison:
        st.caption(
            "같은 이야기 맥락과 분량에서 읽기 프로필을 생성에 쓰지 않은 "
            "기본 LLM 장을 한 번 만들고, 두 결과를 같은 Kiwi·G2P "
            "기준으로 검사합니다."
        )
        if st.button(
            "단순 LLM과 점수 비교",
            icon=":material/compare_arrows:",
            width="stretch",
            key=(
                "dynamic-compare-"
                + str(
                    _as_mapping(
                        current_record.get("response")
                    ).get("generationId", record_index)
                )
            ),
        ):
            with st.status(
                "같은 조건의 기본 LLM 장을 만들고 검사하고 있어요.",
                expanded=True,
            ) as status:
                succeeded = _compare_record(record_index)
                status.update(
                    label=(
                        "장 비교 결과가 준비됐어요."
                        if succeeded
                        else "장 비교에 실패했어요."
                    ),
                    state="complete" if succeeded else "error",
                    expanded=not succeeded,
                )
            if succeeded:
                st.rerun()
        latest_records = st.session_state.dynamic_chapter_history
        latest_record = (
            latest_records[record_index]
            if 0 <= record_index < len(latest_records)
            else current_record
        )
        error = _as_mapping(latest_record).get("comparisonError")
        if error:
            st.error(str(error), icon=":material/error:")
        return

    plain = _as_mapping(comparison.get("plain"))
    personalized = _as_mapping(comparison.get("personalized"))
    plain_fit = _as_mapping(plain.get("fit"))
    personalized_fit = _as_mapping(personalized.get("fit"))
    comparison_result = _as_mapping(comparison.get("comparison"))
    delta = _as_mapping(comparison_result.get("delta"))
    score_delta = _numeric(delta.get("profileFitScore"))
    is_partial = (
        comparison_result.get("comparisonConfidence") == "PARTIAL"
    )
    plain_display_score = (
        comparison_result.get("plainProfileFitScore")
        if is_partial
        else plain_fit.get("profileFitScore")
    )
    personalized_display_score = (
        comparison_result.get("personalizedProfileFitScore")
        if is_partial
        else personalized_fit.get("profileFitScore")
    )

    st.metric(
        "단순 LLM 적합도",
        _score_label(
            plain_display_score,
            "PARTIAL" if is_partial else plain_fit.get("scoreConfidence"),
        ),
        border=True,
    )
    st.metric(
        "아동 프로필 적용",
        _score_label(
            personalized_display_score,
            (
                "PARTIAL"
                if is_partial
                else personalized_fit.get("scoreConfidence")
            ),
        ),
        delta=(
            f"{score_delta:+.1f}점"
            if score_delta is not None
            else None
        ),
        border=True,
    )
    st.caption(
        "0~100 적합도는 임상 점수나 이야기 재미 점수가 아닙니다. "
        "현재 아동 읽기 프로필의 회피·제한 규칙에 대한 개발용 지표입니다."
    )
    st.caption(
        "분석 상태 · 기본 "
        f"{plain_fit.get('analysisStatus', '-')} / 개인화 "
        f"{personalized_fit.get('analysisStatus', '-')}"
    )
    st.caption(
        "검증 범위 · 기본 "
        f"{plain_fit.get('scoreCoveragePercent', '-')}% / 개인화 "
        f"{personalized_fit.get('scoreCoveragePercent', '-')}%"
    )

    if not comparison_result.get("comparable"):
        reason = str(
            comparison_result.get("comparisonReason")
            or plain_fit.get("scoreReason")
            or personalized_fit.get("scoreReason")
            or "Kiwi·G2P 분석 신뢰도가 충분하지 않아 점수를 확정하지 못했습니다."
        )
        st.warning(reason, icon=":material/rule:")
    else:
        if is_partial:
            st.warning(
                str(
                    comparison_result.get("comparisonReason")
                    or "확인 가능한 규칙만 반영한 참고 점수입니다."
                ),
                icon=":material/rule:",
            )
        winner = str(comparison_result.get("winner") or "")
        if is_partial:
            winner_label = {
                "PERSONALIZED": (
                    "확인 가능한 규칙 기준으로 개인화 점수가 더 높아요."
                ),
                "PLAIN": (
                    "확인 가능한 규칙 기준으로 기본 LLM 점수가 더 높아요."
                ),
                "TIE": "확인 가능한 규칙 기준으로 두 점수가 같아요.",
            }.get(winner)
        else:
            winner_label = {
                "PERSONALIZED": "이번 장은 현재 개인화 결과의 적합도가 더 높아요.",
                "PLAIN": "이번 장은 기본 LLM 결과의 적합도가 더 높아요.",
                "TIE": "이번 장은 두 결과의 적합도가 같아요.",
            }.get(winner)
        if winner_label:
            st.info(winner_label, icon=":material/analytics:")

    st.markdown("**계약과 초과량**")
    st.dataframe(
        [
            {
                "항목": "페이지 형식",
                "기본": _contract_label(
                    plain_fit.get("contractPass")
                ),
                "개인화": _contract_label(
                    personalized_fit.get("contractPass")
                ),
            },
            {
                "항목": "확인된 회피 규칙 초과",
                "기본": _occurrence_label(
                    plain_fit.get("excludedOverage")
                ),
                "개인화": _occurrence_label(
                    personalized_fit.get("excludedOverage")
                ),
            },
            {
                "항목": "확인된 제한 규칙 초과",
                "기본": _occurrence_label(
                    plain_fit.get("limitedOverage")
                ),
                "개인화": _occurrence_label(
                    personalized_fit.get("limitedOverage")
                ),
            },
        ],
        hide_index=True,
        width="stretch",
    )

    request = _as_mapping(current_record.get("request"))
    profile = _as_mapping(request.get("generationProfile"))
    skills = profile.get("skills", [])
    plain_occurrences = _as_mapping(
        plain_fit.get("featureOccurrences")
    )
    personalized_occurrences = _as_mapping(
        personalized_fit.get("featureOccurrences")
    )
    if isinstance(skills, list) and skills:
        st.markdown("**읽기 규칙별 검출**")
        st.dataframe(
            [
                {
                    "규칙": skill.get("code", "-"),
                    "정책": skill.get("role", "-"),
                    "기본": _occurrence_label(
                        plain_occurrences.get(
                            str(skill.get("code", ""))
                        )
                    ),
                    "개인화": _occurrence_label(
                        personalized_occurrences.get(
                            str(skill.get("code", ""))
                        )
                    ),
                }
                for skill in skills
                if isinstance(skill, Mapping)
            ],
            hide_index=True,
            width="stretch",
        )

    diagnostics = _as_mapping(comparison.get("diagnostics"))
    plain_timing = _as_mapping(plain.get("timingMs"))
    personalized_timing = _as_mapping(personalized.get("timingMs"))
    with st.container(border=True):
        st.markdown("**속도와 호출 수**")
        st.write(
            "기본 LLM 처리 · "
            f"{_format_ms(plain_timing.get('total'))}"
        )
        st.write(
            "현재 개인화 처리 · "
            f"{_format_ms(personalized_timing.get('total'))}"
        )
        st.write(
            "비교 요청 왕복 · "
            f"{_format_ms(current_record.get('comparisonWallMs'))}"
        )
        st.write(
            "API 호출 · 기본 "
            f"{diagnostics.get('baselineApiCallCount', '-')}회 / "
            "개인화 "
            f"{diagnostics.get('personalizedApiCallCount', '-')}회"
        )
        st.caption(
            "비교 버튼으로 새로 추가된 호출 · "
            f"{diagnostics.get('newApiCallCount', '-')}회"
        )

    with st.expander(
        "두 장의 본문 이어서 보기",
        icon=":material/vertical_split:",
    ):
        st.markdown("### 기본 LLM")
        _render_chapter_text(plain.get("pages"))
        st.divider()
        st.markdown("### 현재 개인화")
        _render_chapter_text(personalized.get("pages"))


def _render_dev_panel(
    response: Mapping[str, Any],
    cursor: ChapterDisplayCursor,
    wall_ms: float,
    image_entry: Mapping[str, Any],
) -> None:
    timing = response.get("timingMs", {})
    generation = response.get("generation", {})
    quality = response.get("quality", {})
    page_qualities = quality.get("pages", [])
    page_quality = (
        page_qualities[cursor.page_index].get("quality", {})
        if cursor.page_index < len(page_qualities)
        else {}
    )
    with st.container(border=True):
        st.subheader("개발자 측정")
        st.metric("전체 왕복", f"{wall_ms / 1000:.2f}초")
        st.metric(
            "글 생성",
            f"{float(timing.get('generation') or 0) / 1000:.2f}초",
        )
        st.metric(
            "Kiwi·G2P 검사",
            f"{float(timing.get('analysis') or 0) / 1000:.2f}초",
        )
        st.metric(
            "페이지 분할",
            f"{float(timing.get('pagination') or 0):.1f}ms",
        )
        st.metric(
            "조건부 교정",
            f"{float(timing.get('repair') or 0) / 1000:.2f}초",
        )
        st.metric(
            "장면 설계",
            f"{float(timing.get('visualScene') or 0) / 1000:.2f}초",
        )
        image_metadata = _as_mapping(image_entry.get("response"))
        image_provider_ms = _numeric(image_metadata.get("timingMs"))
        st.metric(
            "그림 모델 생성",
            _format_ms(image_provider_ms),
        )
        image_progress = _chapter_image_progress(response)
        st.metric(
            "이 장 그림 준비",
            f"{image_progress['ready']} / {image_progress['total']}장",
        )
        st.metric(
            "병렬 그림 작업",
            (
                f"{image_progress['running']}장 실행 · "
                f"{image_progress['queued']}장 대기"
            ),
        )
        st.caption(
            f"장 전체 예약 · 최대 {MAX_PARALLEL_IMAGE_JOBS}장 병렬"
            + (
                f" · 실패 {image_progress['failed']}장"
                if image_progress["failed"]
                else ""
            )
        )
        st.divider()
        st.write(
            f"후보 {generation.get('candidateCount', '-')}개 · "
            f"모델 호출 {generation.get('apiCallCount', '-')}회"
        )
        st.write(
            "모델 공급자: "
            f"{generation.get('provider', '-')} · "
            f"{generation.get('model', '-')}"
        )
        repair_status = (
            "채택"
            if generation.get("repairAccepted")
            else "기각"
            if generation.get("repairAttempted")
            else "미시도"
        )
        st.write(f"국소 교정: {repair_status}")
        st.write(
            f"동적 페이지 수: {generation.get('pageCount', '-')}"
        )
        st.write(
            f"현재 페이지 음절: "
            f"{page_quality.get('writtenSyllableCount', '-')}"
        )
        st.write(
            f"읽기 품질: {page_quality.get('status', '-')}"
        )
        st.write(
            f"분석 상태: {page_quality.get('analysisStatus', '-')}"
        )
        if image_entry:
            st.write(
                "그림 공급자: Gemini · "
                f"{image_metadata.get('model', '-')}"
            )
            st.write(
                "그림 API 왕복: "
                f"{_format_ms(image_entry.get('wallMs'))}"
            )


st.set_page_config(
    page_title="iRead 동적 장 테스트",
    page_icon="📖",
    layout="wide",
)
_initialize_state()

header = st.container(horizontal=True, vertical_alignment="center")
header.title("iRead 이야기·그림 통합 테스트")
header.badge("기본 1회 · 필요 시 교정 1회", color="blue")
header.badge("2~4페이지 자동 분할", color="green")
header.badge("장 전체 예약 · 최대 2장 병렬", color="orange")

server_health = _get_server_health(
    str(st.session_state["dynamic-endpoint"]).strip()
)
story_provider = server_health["storyProvider"].lower()
if story_provider == "mock":
    st.warning(
        "현재 AI 서버가 **mock 모드**입니다. 이야기 제목과 프롬프트를 "
        "실제 모델이 읽지 않으므로 공통 테스트 문장이 나올 수 있습니다. "
        "내용 품질을 확인하려면 서버를 GMS 모드로 다시 시작해 주세요.",
        icon=":material/warning:",
    )
elif story_provider == "gms":
    st.caption("연결 상태 · GMS 경유 GPT 이야기 생성")
elif story_provider == "openai":
    st.caption("연결 상태 · OpenAI 직접 이야기 생성")
else:
    st.warning(
        "AI 서버의 생성 공급자를 확인하지 못했습니다. 연결 설정과 "
        "`/health` 응답을 확인해 주세요.",
        icon=":material/cloud_off:",
    )

with st.sidebar:
    st.subheader("연결 설정")
    st.text_input(
        "장 생성 API",
        key="dynamic-endpoint",
        persist_state="session",
    )
    st.text_input(
        "그림 생성 API",
        key="dynamic-image-endpoint",
        persist_state="session",
    )
    st.text_input(
        "X-API-Key",
        key="dynamic-api-key",
        type="password",
        persist_state="session",
    )
    st.toggle(
        "장 응답 뒤 모든 페이지 그림 자동 예약",
        key="dynamic-auto-image",
        on_change=_on_auto_image_change,
        help=(
            "장 글이 도착하면 2~4페이지 그림을 모두 예약하고 "
            "최대 2장씩 백그라운드에서 병렬 생성합니다."
        ),
    )
    st.caption(
        "글과 페이지 이동은 그림을 기다리지 않습니다. 3·4페이지도 즉시 "
        "대기열에 들어가며 앞선 그림이 끝나는 대로 시작합니다."
    )
    st.caption(
        "현재 장의 그림은 최대 4장만 세션에 보관합니다. "
        "기본 LLM 비교는 오른쪽 개발 패널에서 별도로 실행합니다."
    )
    st.caption(
        "현재 공급자 · "
        + (
            "GMS · GPT"
            if story_provider == "gms"
            else "OpenAI 직접"
            if story_provider == "openai"
            else "목업"
            if story_provider == "mock"
            else "확인 불가"
        )
    )
    if st.session_state.dynamic_story_id is not None:
        if st.button(
            "이야기 다시 고르기",
            icon=":material/restart_alt:",
            width="stretch",
        ):
            _reset_story()
            st.rerun()

if st.session_state.dynamic_story_id is None:
    st.subheader("이야기와 아동 읽기 수준을 골라 주세요")
    selected_story_id = st.selectbox(
        "이야기",
        [story.template_id for story in STORY_CATALOG],
        format_func=lambda template_id: get_story_fixture(
            int(template_id)
        ).title,
        key="dynamic-story-selector",
    )
    selected_profile = st.segmented_control(
        "아동 읽기 수준",
        PROFILE_KEYS,
        format_func=lambda key: READING_PROFILE_PRESETS[key]["label"],
        key="dynamic-profile-selector",
    )
    story = get_story_fixture(int(selected_story_id))
    profile_key = str(selected_profile or "balanced")
    with st.container(border=True):
        st.subheader(story.title)
        st.write(story.description)
        st.caption(
            f"전체 {story.total_chapters}장 · 각 장 2~4페이지 동적 구성"
        )
        st.info(
            READING_PROFILE_PRESETS[profile_key]["description"],
            icon=":material/record_voice_over:",
        )
    if st.button(
        "첫 장 만들기",
        type="primary",
        icon=":material/menu_book:",
        width="stretch",
    ):
        _start_story(story, profile_key)
        if _generate_with_status(story, None):
            st.rerun()
else:
    story = get_story_fixture(int(st.session_state.dynamic_story_id))
    response = st.session_state.dynamic_current_response
    cursor = st.session_state.dynamic_display_cursor
    history = st.session_state.dynamic_chapter_history
    if not isinstance(response, Mapping):
        error = st.session_state.dynamic_last_error
        if error:
            st.error(str(error), icon=":material/error:")
        if st.button(
            "이야기 선택으로 돌아가기",
            icon=":material/arrow_back:",
        ):
            _reset_story()
            st.rerun()
    if isinstance(response, Mapping) and isinstance(
        cursor,
        ChapterDisplayCursor,
    ):
        _harvest_image_jobs()
        current_record = history[-1]
        if bool(st.session_state.get("dynamic-auto-image", True)):
            _enqueue_missing_chapter_images(current_record)
        image_progress = _chapter_image_progress(response)
        main, dev = st.columns([3, 1], gap="large")
        with main:
            page = displayed_page(response, cursor)
            if image_progress["running"] + image_progress["queued"]:
                _render_story_page_polling(
                    story,
                    response,
                    cursor,
                    current_record,
                )
            else:
                _render_story_page_static(
                    story,
                    response,
                    cursor,
                    current_record,
                )
            generation_cursor: ChapterGenerationCursor = (
                st.session_state.dynamic_generation_cursor
            )
            if not cursor.is_last_page:
                if st.button(
                    "다음 페이지",
                    type="primary",
                    icon=":material/arrow_forward:",
                    width="stretch",
                    key=(
                        f"dynamic-next-{cursor.chapter_number}-"
                        f"{cursor.page_number}"
                    ),
                ):
                    st.session_state.dynamic_display_cursor = (
                        advance_display_cursor(cursor)
                    )
                    st.rerun()
            elif generation_cursor.awaiting_branch:
                submitted, branch = _render_answer_form(
                    page,
                    chapter_number=cursor.chapter_number,
                )
                if submitted and branch is not None:
                    if _generate_with_status(story, branch):
                        st.rerun()
            elif generation_cursor.complete:
                st.success(
                    "이야기가 끝났어요. 모든 장이 동적 페이지로 "
                    "구성되었습니다.",
                    icon=":material/celebration:",
                )

            error = st.session_state.dynamic_last_error
            if error:
                st.error(str(error), icon=":material/error:")
                if (
                    st.session_state.dynamic_retryable
                    and st.session_state.dynamic_pending_request
                    and st.button(
                        "같은 요청 다시 보내기",
                        icon=":material/replay:",
                    )
                ):
                    if _generate_with_status(
                        story,
                        st.session_state.dynamic_pending_branch,
                        retry=True,
                    ):
                        st.rerun()

            with st.expander(
                "생성된 장 기록",
                icon=":material/history:",
            ):
                for record in history:
                    chapter_response = record["response"]
                    st.markdown(
                        f"**{chapter_response['chapterNumber']}장 · "
                        f"{len(chapter_response['pages'])}페이지**"
                    )
                    for generated_page in chapter_response["pages"]:
                        st.caption(
                            f"{generated_page['pageNumber']}페이지"
                        )
                        for sentence in generated_page["sentences"]:
                            st.write(sentence)
            with st.expander(
                "마지막 API 요청과 응답",
                icon=":material/code:",
            ):
                st.markdown("**요청**")
                st.json(history[-1]["request"])
                st.markdown("**응답**")
                st.json(history[-1]["response"])
                latest_image = _current_page_image(
                    response,
                    int(page["pageNumber"]),
                )
                if latest_image:
                    st.divider()
                    st.markdown("**현재 페이지 그림 요청**")
                    st.json(latest_image.get("request", {}))
                    st.markdown("**현재 페이지 그림 응답 메타데이터**")
                    st.json(latest_image.get("response", {}))
        with dev:
            _render_dev_panel(
                response,
                cursor,
                float(history[-1]["wallMs"]),
                _current_page_image(
                    response,
                    int(page["pageNumber"]),
                ),
            )
            _render_chapter_comparison_panel(
                history[-1],
                len(history) - 1,
            )
