from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from iread_ai.application.branch_input_review import DeterministicBranchInputReviewer
from iread_ai.config import Settings
from iread_ai.main import create_app

API_PATH = "/api/v1/story/branch-input/review"
API_KEY = "branch-review-integration-key"


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(
        app_env="test",
        internal_api_key=API_KEY,
        story_provider="mock",
        generation_provider="mock",
    )
    with TestClient(
        create_app(
            settings=settings,
            branch_input_reviewer=DeterministicBranchInputReviewer(),
        )
    ) as test_client:
        yield test_client


def payload(transcript: str = "강을 따라가 볼래요") -> dict[str, object]:
    return {
        "requestId": "review-1",
        "question": "토끼는 이제 무엇을 할까요?",
        "options": ["다리를 건너요", "친구를 불러요", "숲으로 돌아가요"],
        "transcript": transcript,
    }


def headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Idempotency-Key": "review-1"}


def test_review_endpoint_returns_compact_structured_decision(client: TestClient) -> None:
    response = client.post(API_PATH, headers=headers(), json=payload())

    assert response.status_code == 200
    assert response.json() == {
        "requestId": "review-1",
        "decision": "ALLOW",
        "reasonCode": "OK",
        "policyVersion": "story-branch-input-v1",
    }


def test_blocked_input_returns_only_reason_code(client: TestClient) -> None:
    response = client.post(
        API_PATH,
        headers=headers(),
        json=payload("목을 잘라 버릴래요"),
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "BLOCK"
    assert response.json()["reasonCode"] == "SEVERE_VIOLENCE"
    assert "transcript" not in response.json()


def test_identical_review_replays_without_changing_decision(client: TestClient) -> None:
    first = client.post(API_PATH, headers=headers(), json=payload())
    replay = client.post(API_PATH, headers=headers(), json=payload())

    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.headers["Idempotent-Replayed"] == "true"


def test_review_requires_internal_api_key(client: TestClient) -> None:
    response = client.post(
        API_PATH,
        headers={"Idempotency-Key": "review-1"},
        json=payload(),
    )

    assert response.status_code == 401
