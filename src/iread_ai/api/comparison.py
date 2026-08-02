from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from iread_ai.api.dependencies import (
    AuthenticatedService,
    require_internal_api_key,
)
from iread_ai.api.errors import ApiError
from iread_ai.application.personalized_chapter_service import (
    StoryChapterUseCaseError,
)
from iread_ai.contracts.comparison import (
    DisplayedChapterComparisonRequest,
)
from iread_ai.personalization.chapter_comparison import (
    new_chapter_comparison_id,
)

router = APIRouter(prefix="/api/dev/story", tags=["development"])
Auth = Annotated[AuthenticatedService, Depends(require_internal_api_key)]


@router.post(
    "/displayed-chapter-comparison",
    response_model=dict[str, Any],
    operation_id="compareDisplayedPersonalizedChapterToPlain",
    summary="실제 표시된 개인화 장을 일반 LLM 한 번 생성과 비교",
)
async def compare_displayed_chapter(
    payload: DisplayedChapterComparisonRequest,
    request: Request,
    _auth: Auth,
) -> dict[str, Any]:
    if request.app.state.settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Not found")

    chapter_request = payload.chapter_request
    personalized = payload.personalized_response
    try:
        result = await (
            request.app.state.chapter_generation_comparison_service
        ).compare_displayed_chapter_to_plain(
            chapter_request,
            personalized,
        )
    except StoryChapterUseCaseError as exc:
        raise ApiError(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            request_id=payload.request_id,
            retryable=exc.retryable,
        ) from exc
    return {
        "requestId": payload.request_id,
        "schemaVersion": 1,
        "comparisonId": new_chapter_comparison_id(),
        "storyId": chapter_request.story_id,
        "storyRevision": chapter_request.story_revision,
        "chapterNumber": chapter_request.chapter_number,
        **result,
    }
