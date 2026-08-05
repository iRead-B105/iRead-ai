from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import iread_ai.app as app_module
from iread_ai.idempotency import MemoryIdempotencyStore

client = TestClient(app_module.app)
AUTH_HEADERS = {"X-API-Key": "test-internal-key"}


@pytest.fixture(autouse=True)
def configured_internal_key(monkeypatch) -> None:
    configured = app_module.settings.model_copy(
        update={"internal_api_key": SecretStr(AUTH_HEADERS["X-API-Key"])}
    )
    monkeypatch.setattr(app_module, "settings", configured)
    monkeypatch.setattr(app_module.app.state, "settings", configured)
    monkeypatch.setattr(
        app_module,
        "idempotency_store",
        MemoryIdempotencyStore(ttl_seconds=60),
    )


def _payload(request_id: str) -> dict[str, object]:
    return {
        "requestId": request_id,
        "trainingId": 1,
        "studentId": 2,
        "trainingTemplateId": 30,
        "schemaVersion": 1,
        "result": {
            "pronunciationAnalyses": [
                {
                    "questionNo": 1,
                    "pronunciationAccuracyScore": 82.5,
                    "fluencyScore": 76,
                    "completenessScore": 95,
                    "attemptNo": 1,
                }
            ]
        },
    }


def test_pronunciation_result_is_evaluated_without_llm() -> None:
    request_id = "training-evaluation-pronunciation"
    response = client.post(
        "/api/v1/trainings/evaluate",
        headers={**AUTH_HEADERS, "Idempotency-Key": request_id},
        json=_payload(request_id),
    )

    assert response.status_code == 200, response.text
    assert response.json()["accuracy"] == 82.5
    assert response.headers["X-AI-Provider"] == "hybrid-evaluator"


def test_training_evaluation_is_idempotent() -> None:
    request_id = "training-evaluation-replay"
    headers = {**AUTH_HEADERS, "Idempotency-Key": request_id}
    payload = _payload(request_id)

    first = client.post("/api/v1/trainings/evaluate", headers=headers, json=payload)
    second = client.post("/api/v1/trainings/evaluate", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert first.json() == second.json()


def test_invalid_pronunciation_score_returns_422() -> None:
    request_id = "training-evaluation-invalid-score"
    payload = _payload(request_id)
    payload["result"]["pronunciationAnalyses"][0]["pronunciationAccuracyScore"] = 101  # type: ignore[index]

    response = client.post(
        "/api/v1/trainings/evaluate",
        headers={**AUTH_HEADERS, "Idempotency-Key": request_id},
        json=payload,
    )

    assert response.status_code == 422
    assert "between 0 and 100" in response.json()["detail"]
