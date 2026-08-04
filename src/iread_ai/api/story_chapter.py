from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from iread_ai.api.dependencies import (
    AuthenticatedService,
    require_idempotency_key,
    require_internal_api_key,
)
from iread_ai.api.errors import ApiError
from iread_ai.application.personalized_chapter_service import (
    StoryChapterUseCaseError,
)
from iread_ai.contracts.common import ErrorResponse
from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.ports.idempotency_store import (
    BeginKind,
    IdempotencyScope,
    IdempotencyStore,
    StoredResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v3/story/chapters", tags=["story-v3"])

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "요청 형식 오류"},
    401: {"model": ErrorResponse, "description": "내부 API 키 인증 실패"},
    409: {"model": ErrorResponse, "description": "멱등키 충돌 또는 처리 중"},
    422: {"model": ErrorResponse, "description": "이야기 상태 또는 정책 오류"},
    502: {"model": ErrorResponse, "description": "AI 모델 출력 또는 연결 오류"},
    504: {"model": ErrorResponse, "description": "AI 모델 처리 시간 초과"},
}
OPENAPI_OPERATION_POLICY = {
    "x-timeout-ms": 40000,
    "x-idempotency-required": True,
    "x-retry-policy": ("연결 실패와 502·503·504에 한해 같은 Idempotency-Key로 최대 1회 재시도"),
}

Auth = Annotated[AuthenticatedService, Depends(require_internal_api_key)]
IdempotencyKey = Annotated[str, Depends(require_idempotency_key)]


@router.post(
    "/generate",
    response_model=StoryChapterGenerateResponse,
    responses=ERROR_RESPONSES,
    operation_id="generatePersonalizedStoryChapter",
    summary="아동 읽기 프로필과 이야기 상태로 동적 길이의 다음 장 생성",
    response_description="검사와 페이지 분할을 마친 2~4페이지 장",
    openapi_extra=OPENAPI_OPERATION_POLICY,
)
async def generate_personalized_story_chapter(
    payload: StoryChapterGenerateRequest,
    request: Request,
    auth: Auth,
    idempotency_key: IdempotencyKey,
) -> StoryChapterGenerateResponse | JSONResponse:
    store: IdempotencyStore = request.app.state.idempotency_store
    fingerprint = _fingerprint(payload)
    scope = IdempotencyScope(
        api_identity=auth.identity,
        method=request.method,
        path=request.url.path,
        key=idempotency_key,
    )
    begin = await store.begin(scope, fingerprint)

    if begin.kind is BeginKind.REPLAY:
        assert begin.response is not None
        return JSONResponse(
            status_code=begin.response.status_code,
            content=begin.response.body,
            headers={"Idempotent-Replayed": "true"},
        )
    if begin.kind is BeginKind.CONFLICT:
        raise ApiError(
            status_code=409,
            code="IDEMPOTENCY_CONFLICT",
            message="같은 Idempotency-Key를 다른 요청 본문에 사용했습니다.",
            request_id=payload.request_id,
            retryable=False,
        )
    if begin.kind is BeginKind.IN_PROGRESS:
        raise ApiError(
            status_code=409,
            code="IDEMPOTENCY_IN_PROGRESS",
            message="같은 장 생성 요청을 이미 처리하고 있습니다.",
            request_id=payload.request_id,
            retryable=True,
        )

    started = time.perf_counter()
    try:
        response = await request.app.state.story_chapter_service.generate(payload)
        response = response.validate_against_request(payload)
        body = response.model_dump(by_alias=True)
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(status_code=200, body=body),
        )
        logger.info(
            (
                "Personalized chapter completed request_id=%s story_id=%s "
                "chapter=%s pages=%s quality=%s elapsed_ms=%.1f"
            ),
            payload.request_id,
            payload.story_id,
            payload.chapter_number,
            len(response.pages),
            response.quality.chapter.status,
            (time.perf_counter() - started) * 1000,
        )
        return response
    except StoryChapterUseCaseError as exc:
        api_error = ApiError(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            request_id=payload.request_id,
            retryable=exc.retryable,
        )
        if exc.retryable:
            await store.release(scope, fingerprint)
            raise api_error from exc
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(
                status_code=exc.status_code,
                body=api_error.body(),
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=api_error.body(),
        )
    except BaseException:
        await store.release(scope, fingerprint)
        raise


def _fingerprint(payload: StoryChapterGenerateRequest) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["router"]
