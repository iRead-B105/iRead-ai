from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse

from iread_ai.contracts.common import ErrorResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    code: str
    message: str
    request_id: str
    retryable: bool | None = None

    def body(self) -> dict[str, Any]:
        return ErrorResponse(
            code=self.code,
            message=self.message,
            requestId=self.request_id,
            retryable=self.retryable,
        ).model_dump(by_alias=True, exclude_none=True)


def error_request_id(request: Request, body: Any = None) -> str:
    if isinstance(body, dict):
        value = body.get("requestId")
        if isinstance(value, str) and value:
            return value
    idempotency_key = request.headers.get("Idempotency-Key")
    if idempotency_key:
        return idempotency_key[:128]
    return str(uuid4())


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = getattr(exc, "body", None)
        errors = exc.errors()
        is_json_error = any(error.get("type") == "json_invalid" for error in errors)
        idempotency_errors = [
            error
            for error in errors
            if tuple(error.get("loc", ())) == ("header", "Idempotency-Key")
        ]
        missing_idempotency = any(error.get("type") == "missing" for error in idempotency_errors)
        invalid_idempotency = bool(idempotency_errors) and not missing_idempotency

        if missing_idempotency:
            code = "MISSING_IDEMPOTENCY_KEY"
            message = "Idempotency-Key 헤더가 필요합니다."
        elif invalid_idempotency:
            code = "INVALID_IDEMPOTENCY_KEY"
            message = "Idempotency-Key는 1자 이상 128자 이하여야 합니다."
        elif is_json_error:
            code = "INVALID_JSON"
            message = "요청 JSON 형식이 올바르지 않습니다."
        else:
            code = "INVALID_REQUEST_FORMAT"
            message = "요청 필드의 형식이 API 계약과 맞지 않습니다."

        response = ErrorResponse(
            code=code,
            message=message,
            requestId=error_request_id(request, body),
            retryable=False,
        )
        return JSONResponse(
            status_code=400,
            content=response.model_dump(by_alias=True, exclude_none=True),
        )

    @app.exception_handler(ResponseValidationError)
    async def handle_response_validation(
        request: Request, _exc: ResponseValidationError
    ) -> JSONResponse:
        request_id = error_request_id(request)
        logger.error("Response contract validation failed request_id=%s", request_id)
        response = ErrorResponse(
            code="INTERNAL_RESPONSE_ERROR",
            message="AI 서버가 올바른 응답을 구성하지 못했습니다.",
            requestId=request_id,
            retryable=False,
        )
        return JSONResponse(
            status_code=500,
            content=response.model_dump(by_alias=True, exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, _exc: Exception) -> JSONResponse:
        request_id = error_request_id(request)
        logger.error("Unexpected server error request_id=%s", request_id)
        response = ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="AI 서버 내부 오류가 발생했습니다.",
            requestId=request_id,
            retryable=False,
        )
        return JSONResponse(
            status_code=500,
            content=response.model_dump(by_alias=True, exclude_none=True),
        )
