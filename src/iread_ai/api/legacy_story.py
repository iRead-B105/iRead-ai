from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from iread_ai.api.errors import ApiError
from iread_ai.application.personalized_chapter_service import (
    StoryChapterUseCaseError,
)
from iread_ai.generation_models import (
    ContinueStoryRequest,
    GenerateStoryRequest,
    GenerateStoryResponse,
)
from iread_ai.ports.idempotency_store import (
    BeginKind,
    IdempotencyScope,
    StoredResponse,
)

logger = logging.getLogger(__name__)
runtime_logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["generation"])


@router.post(
    "/api/v1/story/generate",
    response_model=GenerateStoryResponse,
    summary="이야기 최초 대사 1~5 생성",
)
async def story_generate(
    payload: GenerateStoryRequest,
    request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> GenerateStoryResponse | JSONResponse:
    return await _run_request(
        payload=payload,
        request=request,
        idempotency_key=idempotency_key,
        x_api_key=x_api_key,
        operation="GENERATE",
        generate=lambda: request.app.state.legacy_story_service.generate(payload),
    )


@router.post(
    "/api/v1/story/continue",
    response_model=GenerateStoryResponse,
    summary="분기 선택을 반영한 이야기 대사 6~10 생성",
)
async def story_continue(
    payload: ContinueStoryRequest,
    request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> GenerateStoryResponse | JSONResponse:
    return await _run_request(
        payload=payload,
        request=request,
        idempotency_key=idempotency_key,
        x_api_key=x_api_key,
        operation="CONTINUE",
        generate=lambda: request.app.state.legacy_story_service.continue_story(
            payload
        ),
    )


async def _run_request(
    *,
    payload: GenerateStoryRequest,
    request: Request,
    idempotency_key: str,
    x_api_key: str | None,
    operation: str,
    generate: Callable[[], Awaitable[GenerateStoryResponse]],
) -> GenerateStoryResponse | JSONResponse:
    started = time.perf_counter()
    identity = _authenticate(request, x_api_key, payload.requestId)
    if idempotency_key != payload.requestId:
        raise ApiError(
            status_code=400,
            code="IDEMPOTENCY_KEY_MISMATCH",
            message="Idempotency-Key와 requestId가 일치해야 합니다.",
            request_id=payload.requestId,
            retryable=False,
        )
    fingerprint = _fingerprint(payload)
    scope = IdempotencyScope(
        api_identity=identity,
        method=request.method,
        path=request.url.path,
        key=idempotency_key,
    )
    store = request.app.state.idempotency_store
    begin = await store.begin(scope, fingerprint)
    if begin.kind is BeginKind.REPLAY:
        assert begin.response is not None
        _log_request_status(
            operation=operation,
            payload=payload,
            outcome="REPLAY",
            http_status=begin.response.status_code,
            retryable=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return JSONResponse(
            status_code=begin.response.status_code,
            content=begin.response.body,
            headers={"Idempotent-Replayed": "true"},
        )
    if begin.kind is BeginKind.CONFLICT:
        raise ApiError(
            status_code=409,
            code="IDEMPOTENCY_CONFLICT",
            message="같은 Idempotency-Key를 다른 이야기 요청에 사용했습니다.",
            request_id=payload.requestId,
            retryable=False,
        )
    if begin.kind is BeginKind.IN_PROGRESS:
        raise ApiError(
            status_code=409,
            code="IDEMPOTENCY_IN_PROGRESS",
            message="같은 이야기 생성 요청을 이미 처리하고 있습니다.",
            request_id=payload.requestId,
            retryable=True,
        )
    try:
        response = await generate()
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(
                status_code=200,
                body=response.model_dump(mode="json"),
            ),
        )
        _log_request_status(
            operation=operation,
            payload=payload,
            outcome="SUCCESS",
            http_status=200,
            retryable=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            response=response,
        )
        return response
    except StoryChapterUseCaseError as exception:
        api_error = ApiError(
            status_code=exception.status_code,
            code=exception.code,
            message=exception.message,
            request_id=payload.requestId,
            retryable=exception.retryable,
        )
        _log_request_status(
            operation=operation,
            payload=payload,
            outcome=(
                "TIMEOUT"
                if exception.status_code == 504
                or exception.code == "MODEL_TIMEOUT"
                else "FAILED"
            ),
            http_status=exception.status_code,
            retryable=exception.retryable,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error_code=exception.code,
        )
        if exception.retryable:
            await store.release(scope, fingerprint)
            raise api_error from exception
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(
                status_code=exception.status_code,
                body=api_error.body(),
            ),
        )
        return JSONResponse(
            status_code=exception.status_code,
            content=api_error.body(),
        )
    except ValueError as exception:
        logger.warning(
            "Legacy story mapping failed request_id=%s error_type=%s",
            payload.requestId,
            type(exception).__name__,
        )
        _log_request_status(
            operation=operation,
            payload=payload,
            outcome="FAILED",
            http_status=502,
            retryable=False,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error_code="MODEL_OUTPUT_INVALID",
        )
        api_error = ApiError(
            status_code=502,
            code="MODEL_OUTPUT_INVALID",
            message="모델이 이야기 생성 계약에 맞는 결과를 만들지 못했습니다.",
            request_id=payload.requestId,
            retryable=False,
        )
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(status_code=502, body=api_error.body()),
        )
        return JSONResponse(status_code=502, content=api_error.body())
    except BaseException as exception:
        _log_request_status(
            operation=operation,
            payload=payload,
            outcome="FAILED",
            http_status=500,
            retryable=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error_code=type(exception).__name__,
        )
        await store.release(scope, fingerprint)
        raise


def _authenticate(
    request: Request,
    supplied_key: str | None,
    request_id: str,
) -> str:
    expected_key = request.app.state.settings.internal_api_key.get_secret_value()
    if supplied_key is None or not hmac.compare_digest(supplied_key, expected_key):
        raise ApiError(
            status_code=401,
            code="INVALID_API_KEY",
            message="내부 API 키 인증에 실패했습니다.",
            request_id=request_id,
            retryable=False,
        )
    return hashlib.sha256(supplied_key.encode("utf-8")).hexdigest()[:16]


def _fingerprint(payload: GenerateStoryRequest) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _log_request_status(
    *,
    operation: str,
    payload: GenerateStoryRequest,
    outcome: str,
    http_status: int,
    retryable: bool,
    elapsed_ms: float,
    error_code: str | None = None,
    response: GenerateStoryResponse | None = None,
) -> None:
    event: dict[str, object] = {
        "event": "story_generation_request",
        "logSchemaVersion": 1,
        "operation": operation,
        "outcome": outcome,
        "requestId": payload.requestId,
        "storyId": payload.storyId,
        "httpStatus": http_status,
        "retryable": retryable,
        "elapsedMs": round(elapsed_ms, 1),
    }
    if error_code is not None:
        event["errorCode"] = error_code
    if response is not None:
        event.update(
            {
                "nextProgress": response.nextProgress,
                "completed": response.completed,
                "lineCount": len(response.lines),
            }
        )
    runtime_logger.info(
        "%s",
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


__all__ = ["router"]
