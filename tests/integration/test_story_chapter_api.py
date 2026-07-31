from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from iread_ai.application.personalized_chapter_service import (
    PersonalizedStoryChapterService,
)
from iread_ai.config import Settings
from iread_ai.main import create_app
from tests.unit.test_personalized_chapter_service import (
    DeterministicAnalyzer,
    RecordingChapterGenerator,
    chapter_candidate,
)
from tests.unit.test_story_chapter_contracts import request_payload

API_PATH = "/api/v3/story/chapters/generate"
API_KEY = "story-chapter-integration-secret"
IDEMPOTENCY_KEY = "idem-story-101-chapter-2"


def _headers(
    *,
    api_key: str = API_KEY,
    idempotency_key: str = IDEMPOTENCY_KEY,
) -> dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Idempotency-Key": idempotency_key,
    }


@pytest.fixture
def chapter_generator() -> RecordingChapterGenerator:
    return RecordingChapterGenerator(chapter_candidate())


@pytest.fixture
def client(
    chapter_generator: RecordingChapterGenerator,
) -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        internal_api_key=API_KEY,
        story_provider="mock",
        idempotency_ttl_seconds=60,
    )
    service = PersonalizedStoryChapterService(
        generator=chapter_generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )
    with TestClient(
        create_app(
            settings=settings,
            story_chapter_service=service,
        )
    ) as test_client:
        yield test_client


def test_v3_requires_internal_api_key_without_model_call(
    client: TestClient,
    chapter_generator: RecordingChapterGenerator,
) -> None:
    response = client.post(
        API_PATH,
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
        json=request_payload(),
    )

    assert response.status_code == 401
    assert chapter_generator.calls == []


def test_v3_returns_all_dynamic_pages_from_one_model_call(
    client: TestClient,
    chapter_generator: RecordingChapterGenerator,
) -> None:
    response = client.post(
        API_PATH,
        headers=_headers(),
        json=request_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == 3
    assert len(body["pages"]) == 2
    assert body["generation"]["pageCount"] == 2
    assert body["generation"]["apiCallCount"] == 1
    assert body["pages"][0]["requiresBranchInput"] is False
    assert body["pages"][-1]["requiresBranchInput"] is True
    assert len(chapter_generator.calls) == 1
    assert "studentId" not in response.text
    assert "RAW_OUTPUT_MUST_NOT_ESCAPE" not in response.text
    assert "STUDENT_PRIVATE_DATA" not in response.text


def test_identical_v3_request_replays_without_second_model_call(
    client: TestClient,
    chapter_generator: RecordingChapterGenerator,
) -> None:
    first = client.post(
        API_PATH,
        headers=_headers(),
        json=request_payload(),
    )
    replay = client.post(
        API_PATH,
        headers=_headers(),
        json=request_payload(),
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert len(chapter_generator.calls) == 1


def test_same_v3_key_with_changed_body_returns_conflict(
    client: TestClient,
    chapter_generator: RecordingChapterGenerator,
) -> None:
    first = client.post(
        API_PATH,
        headers=_headers(),
        json=request_payload(),
    )
    changed = request_payload()
    changed["storyRevision"] = 9
    conflict = client.post(
        API_PATH,
        headers=_headers(),
        json=changed,
    )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(chapter_generator.calls) == 1
