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
from iread_ai.contracts.common import ErrorResponse
from iread_ai.contracts.teacher_report import (
    TeacherReportAnalyzeRequest,
    TeacherReportAnalyzeResponse,
)
from iread_ai.ports.idempotency_store import (
    BeginKind,
    IdempotencyScope,
    IdempotencyStore,
    StoredResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/reports", tags=["teacher-reports-v1"])

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "요청 계약 오류"},
    401: {"model": ErrorResponse, "description": "내부 API 인증 실패"},
    409: {"model": ErrorResponse, "description": "멱등키 충돌 또는 처리 중"},
    500: {"model": ErrorResponse, "description": "분석 처리 오류"},
}
OPENAPI_OPERATION_POLICY = {
    "x-timeout-ms": 30000,
    "x-idempotency-required": True,
    "x-retry-policy": "연결 실패와 500 응답은 같은 Idempotency-Key로 최대 1회 재시도",
}

Auth = Annotated[AuthenticatedService, Depends(require_internal_api_key)]
IdempotencyKey = Annotated[str, Depends(require_idempotency_key)]


@router.post(
    "/analyze",
    response_model=TeacherReportAnalyzeResponse,
    responses=ERROR_RESPONSES,
    operation_id="analyzeTeacherReport",
    summary="학습 프로필과 시선 추세를 교수자용 관찰 문장으로 분석",
    response_description="진단이 아닌 근거 기반 학습 관찰 요약",
    openapi_extra=OPENAPI_OPERATION_POLICY,
)
async def analyze_teacher_report(
    payload: TeacherReportAnalyzeRequest,
    request: Request,
    auth: Auth,
    idempotency_key: IdempotencyKey,
) -> TeacherReportAnalyzeResponse | JSONResponse:
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
            message="같은 Idempotency-Key가 다른 요청 본문에 사용되었습니다.",
            request_id=payload.request_id,
            retryable=False,
        )
    if begin.kind is BeginKind.IN_PROGRESS:
        raise ApiError(
            status_code=409,
            code="IDEMPOTENCY_IN_PROGRESS",
            message="같은 교수자 보고서 분석 요청이 이미 처리 중입니다.",
            request_id=payload.request_id,
            retryable=True,
        )

    started = time.perf_counter()
    try:
        response = await request.app.state.teacher_report_service.analyze(payload)
        response = response.validate_against_request(payload)
        body = response.model_dump(by_alias=True)
        await store.complete(
            scope,
            fingerprint,
            StoredResponse(status_code=200, body=body),
        )
        logger.info(
            ("Teacher report completed request_id=%s provider=%s sufficiency=%s elapsed_ms=%.1f"),
            payload.request_id,
            response.summary_provider,
            response.data_sufficiency,
            (time.perf_counter() - started) * 1000,
        )
        return response
    except BaseException:
        await store.release(scope, fingerprint)
        raise


def _fingerprint(payload: TeacherReportAnalyzeRequest) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["router"]
