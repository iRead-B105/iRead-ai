from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from iread_ai.config import Settings
from iread_ai.main import create_app
from tests.unit.test_story_chapter_contracts import (
    request_payload as story_chapter_request_payload,
)
from tests.unit.test_story_chapter_contracts import (
    response_payload as story_chapter_response_payload,
)

API_PATH = "/api/dev/story/displayed-chapter-comparison"
API_KEY = "comparison-test-key"


class RecordingChapterComparisonService:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    async def compare_displayed_chapter_to_plain(
        self,
        chapter_request: Any,
        personalized_response: Any,
    ) -> dict[str, Any]:
        self.calls.append((chapter_request, personalized_response))
        return {
            "comparisonVersion": "displayed-chapter-vs-plain-v1",
            "plain": {"status": "SUCCESS"},
            "personalized": {"status": "SUCCESS"},
            "comparison": {
                "comparable": True,
                "winner": "PERSONALIZED",
            },
            "diagnostics": {
                "baselineApiCallCount": 1,
                "personalizedApiCallCount": 2,
                "newApiCallCount": 1,
            },
        }


def _app(
    service: RecordingChapterComparisonService,
    *,
    app_env: str = "test",
):
    is_production = app_env == "production"
    return create_app(
        settings=Settings(
            app_env=app_env,
            internal_api_key=(
                API_KEY if not is_production else "production-test-key"
            ),
            story_provider="openai" if is_production else "mock",
            openai_api_key="unused-test-key" if is_production else None,
            generation_provider="gms" if is_production else "mock",
            story_image_provider="disabled",
            gms_key="unused-gms-test-key" if is_production else None,
        ),
        chapter_generation_comparison_service=service,
    )


def _payload() -> dict[str, Any]:
    return {
        "requestId": "displayed-chapter-api-test",
        "chapterRequest": story_chapter_request_payload(),
        "personalizedResponse": story_chapter_response_payload(),
    }


def test_displayed_chapter_endpoint_uses_exact_v3_pair() -> None:
    service = RecordingChapterComparisonService()
    response = TestClient(_app(service)).post(
        API_PATH,
        headers={"X-API-Key": API_KEY},
        json=_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    request_document = story_chapter_request_payload()
    assert body["requestId"] == "displayed-chapter-api-test"
    assert body["storyId"] == request_document["storyId"]
    assert body["chapterNumber"] == request_document["chapterNumber"]
    assert body["comparison"]["winner"] == "PERSONALIZED"
    assert body["diagnostics"]["newApiCallCount"] == 1
    assert len(service.calls) == 1


def test_displayed_chapter_endpoint_requires_internal_api_key() -> None:
    response = TestClient(_app(RecordingChapterComparisonService())).post(
        API_PATH,
        json=_payload(),
    )

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_API_KEY"


def test_displayed_chapter_endpoint_is_hidden_in_production() -> None:
    service = RecordingChapterComparisonService()
    response = TestClient(_app(service, app_env="production")).post(
        API_PATH,
        headers={"X-API-Key": "production-test-key"},
        json=_payload(),
    )

    assert response.status_code == 404
    assert service.calls == []


def test_displayed_chapter_endpoint_rejects_mismatched_revision() -> None:
    service = RecordingChapterComparisonService()
    payload = _payload()
    personalized = payload["personalizedResponse"]
    assert isinstance(personalized, dict)
    personalized["storyRevision"] += 1

    response = TestClient(_app(service)).post(
        API_PATH,
        headers={"X-API-Key": API_KEY},
        json=payload,
    )

    assert response.status_code == 400
    assert service.calls == []
