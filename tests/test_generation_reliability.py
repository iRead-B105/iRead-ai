import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import iread_ai.app as app_module
from iread_ai.idempotency import IdempotencyConflict, MemoryIdempotencyStore
from iread_ai.providers import GenerationProviderError, GMSTextProvider


@pytest.fixture(autouse=True)
def configured_internal_key(monkeypatch) -> None:
    configured = app_module.settings.model_copy(
        update={"internal_api_key": SecretStr("local-development-key")}
    )
    monkeypatch.setattr(app_module, "settings", configured)
    monkeypatch.setattr(app_module.app.state, "settings", configured)


def _headers(key: str) -> dict[str, str]:
    return {
        "X-API-Key": "local-development-key",
        "Idempotency-Key": key,
    }


def _training_request(request_id: str = "reliable-training") -> dict:
    return {
        "requestId": request_id,
        "schemaVersion": 2,
        "trainingType": "SENTENCE_READING",
        "count": 5,
        "difficulty": 2,
        "targetFeatures": [],
        "excludedFeatures": [],
        "additionalPrompt": "",
        "outputTemplate": {
            "data": [{"sentence": "<string>", "tokens": ["<string>"]}]
        },
    }


class FailingTextProvider:
    model = "gpt-5.4-mini"

    def generate_json(self, **_: object) -> dict:
        raise GenerationProviderError("provider unavailable", retryable=True)


def test_training_provider_failure_returns_safe_fallback(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "text_provider", FailingTextProvider())
    monkeypatch.setattr(
        app_module,
        "idempotency_store",
        MemoryIdempotencyStore(ttl_seconds=60),
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("fallback-training"),
        json=_training_request("fallback-training"),
    )

    assert response.status_code == 200
    assert response.headers["X-AI-Provider"] == "curated-fallback"
    assert response.headers["X-AI-Fallback"] == "curated-fallback"
    assert len(response.json()["data"]) == 5


def test_training_idempotent_replay_and_body_conflict(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "text_provider", None)
    monkeypatch.setattr(
        app_module,
        "idempotency_store",
        MemoryIdempotencyStore(ttl_seconds=60),
    )
    client = TestClient(app_module.app)
    request = _training_request("same-key")

    first = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("same-key"),
        json=request,
    )
    replay = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("same-key"),
        json=request,
    )
    changed = {**request, "difficulty": 3}
    conflict = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("same-key"),
        json=changed,
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replayed"] == "true"
    assert replay.json() == first.json()
    assert conflict.status_code == 409


def test_training_generation_requires_both_internal_headers() -> None:
    client = TestClient(app_module.app)
    request = _training_request("auth-training")

    assert client.post(
        "/api/v1/trainings/candidates",
        json=request,
    ).status_code == 401
    assert client.post(
        "/api/v1/trainings/candidates",
        headers={"X-API-Key": "local-development-key"},
        json=request,
    ).status_code == 400

    too_long = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("x" * 257),
        json=request,
    )
    assert too_long.status_code == 400


def test_memory_idempotency_releases_failed_action() -> None:
    store = MemoryIdempotencyStore(ttl_seconds=60)

    with pytest.raises(RuntimeError):
        store.execute(
            scope="test",
            key="key",
            payload={"value": 1},
            action=lambda: (_ for _ in ()).throw(RuntimeError("failed")),
        )

    value, replayed = store.execute(
        scope="test",
        key="key",
        payload={"value": 2},
        action=lambda: "ok",
    )
    assert value == "ok"
    assert replayed is False

    with pytest.raises(IdempotencyConflict):
        store.execute(
            scope="test",
            key="key",
            payload={"value": 3},
            action=lambda: "never",
        )


def test_gms_text_provider_uses_responses_json_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/responses")
        assert request.headers["Authorization"] == "Bearer gms-key"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.4-mini"
        assert payload["text"]["format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={"status": "completed", "output_text": '{"value":"안전한 훈련"}'},
        )

    provider = GMSTextProvider(
        api_key="gms-key",
        model="gpt-5.4-mini",
        base_url="https://gms.example/v1",
        timeout_seconds=1,
        max_output_tokens=256,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.generate_json(
        schema_name="test_schema",
        schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        system_prompt="safe",
        user_prompt="training",
    ) == {"value": "안전한 훈련"}


def test_training_candidates_allow_generic_generation_without_weak_profiles(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "text_provider", None)
    monkeypatch.setattr(
        app_module,
        "idempotency_store",
        MemoryIdempotencyStore(ttl_seconds=60),
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/api/v1/trainings/candidates",
        headers=_headers("generic-training"),
        json=_training_request("generic-training"),
    )

    assert response.status_code == 200
    assert len(response.json()["data"]) == 5


def test_training_evaluation_matches_backend_rule(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "idempotency_store",
        MemoryIdempotencyStore(ttl_seconds=60),
    )
    client = TestClient(app_module.app)
    response = client.post(
        "/api/v1/trainings/evaluate",
        headers=_headers("evaluate-1"),
        json={
            "requestId": "evaluate-1",
            "trainingId": 1,
            "studentId": 2001,
            "trainingTemplateId": 2,
            "schemaVersion": 1,
            "result": {
                "questions": [
                    {"isCorrect": True},
                    {"isCorrect": False},
                    {"isCorrect": True},
                ]
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["accuracy"] == 66.67
