from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, Request, Security
from fastapi.security import APIKeyHeader

from iread_ai.api.errors import ApiError

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="apiKeyAuth",
    auto_error=False,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedService:
    identity: str


async def require_internal_api_key(
    request: Request,
    supplied_key: str | None = Security(api_key_header),
) -> AuthenticatedService:
    expected_key = request.app.state.settings.internal_api_key.get_secret_value()
    if supplied_key is None or not secrets.compare_digest(supplied_key, expected_key):
        raise ApiError(
            status_code=401,
            code="INVALID_API_KEY",
            message="내부 API 키 인증에 실패했습니다.",
            request_id=request.headers.get("Idempotency-Key", "unknown")[:128],
            retryable=False,
        )
    identity = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest()[:16]
    return AuthenticatedService(identity=identity)


async def require_idempotency_key(
    value: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1, max_length=128),
    ],
) -> str:
    if not value.strip():
        raise ApiError(
            status_code=400,
            code="MISSING_IDEMPOTENCY_KEY",
            message="Idempotency-Key 헤더가 필요합니다.",
            request_id="unknown",
            retryable=False,
        )
    return value
