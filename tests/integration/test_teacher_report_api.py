from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from iread_ai.application.reading_profile_request_adapter import (
    build_teacher_report_request,
)
from iread_ai.config import Settings
from iread_ai.contracts.reading_profile import StudentReadingProfileSnapshot
from iread_ai.devtools.backend_profile_samples import backend_profile_sample
from iread_ai.main import create_app
from tests.unit.test_teacher_report_contracts import teacher_report_request_payload

API_PATH = "/api/v1/reports/analyze"
API_KEY = "teacher-report-integration-key"
IDEMPOTENCY_KEY = "teacher-report-idem-1"


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        internal_api_key=API_KEY,
        story_provider="mock",
        generation_provider="mock",
        gms_key=None,
        idempotency_ttl_seconds=60,
    )
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


def _headers(
    *,
    idempotency_key: str = IDEMPOTENCY_KEY,
) -> dict[str, str]:
    return {
        "X-API-Key": API_KEY,
        "Idempotency-Key": idempotency_key,
    }


def test_teacher_report_api_returns_existing_snapshot_fields(
    client: TestClient,
) -> None:
    response = client.post(
        API_PATH,
        headers=_headers(),
        json=teacher_report_request_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requestId"] == "teacher-report-2026-08-03-1"
    assert body["schemaVersion"] == 1
    assert body["analysisVersion"] == "TEACHER_REPORT_ANALYSIS_V1"
    assert body["summaryProvider"] == "deterministic"
    assert body["dataSufficiency"] == "SUFFICIENT"
    assert body["improvedPatterns"]
    assert body["persistentDifficultyPatterns"]
    assert body["gazeDescriptions"]["training"]
    assert body["gazeDescriptions"]["test"]


def test_teacher_report_api_accepts_backend_profile_snapshot_adapter_output(
    client: TestClient,
) -> None:
    sample = backend_profile_sample()
    snapshot = StudentReadingProfileSnapshot.model_validate(
        {"featureProfiles": sample["featureProfiles"]}
    )
    request_id = "backend-profile-teacher-api"
    request = build_teacher_report_request(
        request_id=request_id,
        snapshot=snapshot,
        feature_labels=sample["featureLabels"],
        gaze_trend=sample["gazeTrend"],
    )

    response = client.post(
        API_PATH,
        headers=_headers(idempotency_key=request_id),
        json=request.model_dump(mode="json", by_alias=True),
    )

    assert response.status_code == 200, response.text
    assert response.json()["requestId"] == request_id
    assert response.json()["persistentDifficultyPatterns"]


def test_teacher_report_api_requires_internal_api_key(client: TestClient) -> None:
    response = client.post(
        API_PATH,
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
        json=teacher_report_request_payload(),
    )

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_API_KEY"


def test_teacher_report_api_replays_identical_request(client: TestClient) -> None:
    first = client.post(
        API_PATH,
        headers=_headers(),
        json=teacher_report_request_payload(),
    )
    replay = client.post(
        API_PATH,
        headers=_headers(),
        json=teacher_report_request_payload(),
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.headers["Idempotent-Replayed"] == "true"


def test_teacher_report_api_rejects_changed_body_with_same_key(
    client: TestClient,
) -> None:
    first = client.post(
        API_PATH,
        headers=_headers(),
        json=teacher_report_request_payload(),
    )
    changed = teacher_report_request_payload()
    changed["featureProfiles"][0]["accuracyRate"] = 0.81
    conflict = client.post(API_PATH, headers=_headers(), json=changed)

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_teacher_report_api_rejects_direct_student_identifier(
    client: TestClient,
) -> None:
    payload = teacher_report_request_payload()
    payload["studentId"] = 42
    response = client.post(API_PATH, headers=_headers(), json=payload)

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST_FORMAT"
