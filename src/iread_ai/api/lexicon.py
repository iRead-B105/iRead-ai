from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from iread_ai.api.dependencies import AuthenticatedService, require_internal_api_key
from iread_ai.api.errors import ApiError
from iread_ai.lexicon.contracts import (
    LexiconPaletteRequest,
    LexiconPaletteResponse,
    LexiconStatusResponse,
)
from iread_ai.lexicon.service import LexiconUnavailableError

router = APIRouter(prefix="/api/v1/lexicon", tags=["lexicon-v1"])
Auth = Annotated[AuthenticatedService, Depends(require_internal_api_key)]


@router.get(
    "/status",
    response_model=LexiconStatusResponse,
    operation_id="getLexiconStatus",
    summary="AI 서버 어휘·발음·음운 DB 상태 확인",
)
async def get_lexicon_status(request: Request, _auth: Auth) -> LexiconStatusResponse:
    return request.app.state.lexicon_service.status()


@router.post(
    "/palettes/query",
    response_model=LexiconPaletteResponse,
    operation_id="queryLexiconPalette",
    summary="아동 읽기 정책에 맞는 안전 어휘 팔레트 조회",
)
async def query_lexicon_palette(
    payload: LexiconPaletteRequest,
    request: Request,
    _auth: Auth,
) -> LexiconPaletteResponse:
    try:
        return request.app.state.lexicon_service.build_palette(payload)
    except LexiconUnavailableError as exception:
        raise ApiError(
            status_code=503,
            code="LEXICON_UNAVAILABLE",
            message="어휘·발음·음운 DB를 사용할 수 없습니다.",
            request_id=payload.requestId,
            retryable=False,
        ) from exception


__all__ = ["router"]
