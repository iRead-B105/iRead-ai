from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import iread_ai.app as app_module
from iread_ai.devtools.curriculum_samples import curriculum_sample

client = TestClient(app_module.app)
AUTH_HEADERS = {"X-API-Key": "test-internal-key"}


@pytest.fixture(autouse=True)
def configured_internal_key(monkeypatch) -> None:
    configured = app_module.settings.model_copy(
        update={"internal_api_key": SecretStr(AUTH_HEADERS["X-API-Key"])}
    )
    monkeypatch.setattr(app_module, "settings", configured)
    monkeypatch.setattr(app_module.app.state, "settings", configured)


def test_curriculum_recommendation_endpoint_returns_stage_gated_five() -> None:
    sample = curriculum_sample("자모 읽기가 어려운 학생")
    request_id = "curriculum-api-letter-stage"
    response = client.post(
        "/api/v1/curricula/recommend",
        headers={**AUTH_HEADERS, "Idempotency-Key": request_id},
        json={
            "requestId": request_id,
            "schemaVersion": 1,
            "featureProfiles": sample["featureProfiles"],
            "recentTrainings": sample["recentTrainings"],
            "useLlm": False,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["currentStage"] == 1
    assert body["maximumAllowedStage"] == 2
    assert len(body["recommendations"]) == 5
    assert all(item["curriculumStage"] <= 2 for item in body["recommendations"])
    assert response.headers["X-AI-Provider"] == "deterministic"


def test_curriculum_recommendation_is_idempotent() -> None:
    sample = curriculum_sample("음절 결합이 어려운 학생")
    request_id = "curriculum-api-idempotent"
    payload = {
        "requestId": request_id,
        "schemaVersion": 1,
        "featureProfiles": sample["featureProfiles"],
        "recentTrainings": sample["recentTrainings"],
        "useLlm": False,
    }
    headers = {**AUTH_HEADERS, "Idempotency-Key": request_id}

    first = client.post("/api/v1/curricula/recommend", headers=headers, json=payload)
    second = client.post("/api/v1/curricula/recommend", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert second.headers["Idempotent-Replayed"] == "true"


def test_curriculum_recommendation_rejects_missing_api_key() -> None:
    response = client.post(
        "/api/v1/curricula/recommend",
        headers={"Idempotency-Key": "curriculum-no-auth"},
        json={
            "requestId": "curriculum-no-auth",
            "schemaVersion": 1,
            "featureProfiles": [],
            "recentTrainings": [],
            "useLlm": False,
        },
    )

    assert response.status_code == 401
