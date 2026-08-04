from __future__ import annotations

import hashlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from iread_ai.api.dependencies import (
    AuthenticatedService,
    require_idempotency_key,
    require_internal_api_key,
)
from iread_ai.api.errors import ApiError
from iread_ai.application.branch_input_review import BranchInputReviewProviderError
from iread_ai.contracts.branch_input_review import (
    BranchInputReviewRequest,
    BranchInputReviewResponse,
)
from iread_ai.contracts.common import ErrorResponse
from iread_ai.ports.idempotency_store import (
    BeginKind,
    IdempotencyScope,
    StoredResponse,
)

router = APIRouter(prefix="/api/v1/story/branch-input", tags=["story"])
Auth = Annotated[AuthenticatedService, Depends(require_internal_api_key)]
IdempotencyKey = Annotated[str, Depends(require_idempotency_key)]


@router.post(
    "/review",
    response_model=BranchInputReviewResponse,
    operation_id="reviewStoryBranchInput",
    summary="STT 분기 입력의 아동 안전성과 현재 질문 관련성 판정",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def review_story_branch_input(
    payload: BranchInputReviewRequest,
    request: Request,
    _auth: Auth,
    idempotency_key: IdempotencyKey,
) -> BranchInputReviewResponse | JSONResponse:
    if idempotency_key != payload.request_id:
        raise ApiError(
            status_code=400,
            code="IDEMPOTENCY_KEY_MISMATCH",
            message="Idempotency-Key와 requestId가 일치해야 합니다.",
            request_id=payload.request_id,
            retryable=False,
        )
    canonical = json.dumps(
        payload.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    scope = IdempotencyScope(
        api_identity=_auth.identity,
        method=request.method,
        path=request.url.path,
        key=idempotency_key,
    )
    store = request.app.state.idempotency_store
    begin = await store.begin(scope, fingerprint)
    if begin.kind is BeginKind.REPLAY:
        assert begin.response is not None
        return JSONResponse(
            status_code=begin.response.status_code,
            content=begin.response.body,
            headers={"Idempotent-Replayed": "true"},
        )
    if begin.kind is not BeginKind.NEW:
        raise ApiError(
            status_code=409,
            code=(
                "IDEMPOTENCY_CONFLICT"
                if begin.kind is BeginKind.CONFLICT
                else "IDEMPOTENCY_IN_PROGRESS"
            ),
            message="같은 검토 요청을 이미 처리 중이거나 다른 본문에 사용했습니다.",
            request_id=payload.request_id,
            retryable=begin.kind is BeginKind.IN_PROGRESS,
        )
    try:
        result = await request.app.state.branch_input_reviewer.review(payload)
        body = result.model_dump(mode="json", by_alias=True)
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(status_code=200, body=body),
        )
        return result
    except BranchInputReviewProviderError as exc:
        await store.release(scope, fingerprint)
        raise ApiError(
            status_code=504 if exc.timeout else 502,
            code="BRANCH_REVIEW_TIMEOUT" if exc.timeout else "BRANCH_REVIEW_FAILED",
            message="분기 입력 안전성 검토를 완료하지 못했습니다.",
            request_id=payload.request_id,
            retryable=exc.retryable,
        ) from exc
    except BaseException:
        await store.release(scope, fingerprint)
        raise


__all__ = ["router"]
