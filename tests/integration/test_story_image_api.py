from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from iread_ai.application.story_image_service import (
    StoryImageApplicationService,
)
from iread_ai.config import Settings
from iread_ai.main import create_app
from tests.unit.test_story_image_contracts import image_request_payload
from tests.unit.test_story_image_service import (
    EmptyReferenceRepository,
    RecordingImageGenerator,
)

API_PATH = "/api/v1/story/images/generate"
API_KEY = "story-image-integration-key"
IDEMPOTENCY_KEY = "story-image-idem-1"


@pytest.fixture
def generator() -> RecordingImageGenerator:
    return RecordingImageGenerator()


@pytest.fixture
def client(
    generator: RecordingImageGenerator,
) -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        internal_api_key=API_KEY,
        story_provider="mock",
        story_image_provider="disabled",
        generation_provider="mock",
        gms_key=None,
        idempotency_ttl_seconds=60,
    )
    service = StoryImageApplicationService(
        generator=generator,
        references=EmptyReferenceRepository(),
    )
    with TestClient(
        create_app(
            settings=settings,
            story_image_service=service,
        )
    ) as test_client:
        yield test_client


def _payload() -> dict[str, object]:
    payload = image_request_payload()
    payload["characterReferences"] = []
    return payload


def _headers(
    *,
    idempotency_key: str = IDEMPOTENCY_KEY,
) -> dict[str, str]:
    return {
        "X-API-Key": API_KEY,
        "Idempotency-Key": idempotency_key,
    }


def test_image_api_requires_internal_api_key_without_model_call(
    client: TestClient,
    generator: RecordingImageGenerator,
) -> None:
    response = client.post(
        API_PATH,
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
        json=_payload(),
    )

    assert response.status_code == 401
    assert generator.calls == []


def test_image_api_returns_base64_image_and_generation_metadata(
    client: TestClient,
    generator: RecordingImageGenerator,
) -> None:
    response = client.post(
        API_PATH,
        headers=_headers(),
        json=_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == "image-page-101-2-1"
    assert body["schemaVersion"] == 1
    assert body["mimeType"] == "image/png"
    assert base64.b64decode(body["imageBase64"]).startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert body["model"] == "gemini-2.5-flash-image"
    assert body["promptVersion"].startswith("story-image-")
    assert body["timingMs"] >= 0
    assert len(generator.calls) == 1


def test_identical_image_request_replays_without_second_model_call(
    client: TestClient,
    generator: RecordingImageGenerator,
) -> None:
    first = client.post(API_PATH, headers=_headers(), json=_payload())
    replay = client.post(API_PATH, headers=_headers(), json=_payload())

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert len(generator.calls) == 1


def test_same_image_key_with_changed_body_returns_conflict(
    client: TestClient,
    generator: RecordingImageGenerator,
) -> None:
    first = client.post(API_PATH, headers=_headers(), json=_payload())
    changed = _payload()
    changed["pageNumber"] = 2
    conflict = client.post(API_PATH, headers=_headers(), json=changed)

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(generator.calls) == 1


def test_client_supplied_image_data_is_rejected_before_model_call(
    client: TestClient,
    generator: RecordingImageGenerator,
) -> None:
    payload = _payload()
    payload["previousImageBase64"] = "iVBORw0KGgo="
    response = client.post(API_PATH, headers=_headers(), json=payload)

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST_FORMAT"
    assert generator.calls == []


def test_unconfigured_endpoint_reports_503() -> None:
    settings = Settings(
        app_env="test",
        internal_api_key=API_KEY,
        story_provider="mock",
        story_image_provider="disabled",
        generation_provider="mock",
        gms_key=None,
    )
    with TestClient(create_app(settings=settings)) as client:
        response = client.post(
            API_PATH,
            headers=_headers(),
            json=_payload(),
        )

    assert response.status_code == 503
    assert response.json()["code"] == "IMAGE_PROVIDER_NOT_CONFIGURED"
