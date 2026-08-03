from __future__ import annotations

import asyncio
import math

import pytest

from iread_ai.adapters.idempotency.memory import MemoryIdempotencyStore
from iread_ai.ports.idempotency_store import (
    BeginKind,
    IdempotencyScope,
    StoredResponse,
)


def _scope(
    *,
    api_identity: str = "spring-backend",
    method: str = "POST",
    path: str = "/api/v1/story/turns",
    key: str = "16d1061f-c9d9-4ff0-8fd5-39cdadbccf71",
) -> IdempotencyScope:
    return IdempotencyScope(
        api_identity=api_identity,
        method=method,
        path=path,
        key=key,
    )


def test_begin_is_atomic_for_concurrent_identical_requests() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        results = await asyncio.gather(*[store.begin(_scope(), "fingerprint-a") for _ in range(32)])

        assert [result.kind for result in results].count(BeginKind.NEW) == 1
        assert [result.kind for result in results].count(BeginKind.IN_PROGRESS) == 31

    asyncio.run(scenario())


def test_completed_response_is_replayed_with_exact_status_and_body() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        body = {
            "schemaVersion": 1,
            "requestId": "16d1061f-c9d9-4ff0-8fd5-39cdadbccf71",
            "error": {
                "code": "PROVIDER_TIMEOUT",
                "retryable": True,
            },
        }

        assert (await store.begin(_scope(), "fingerprint-a")).kind is (BeginKind.NEW)
        assert await store.complete(
            _scope(),
            "fingerprint-a",
            StoredResponse(status_code=504, body=body),
        )

        replay = await store.begin(_scope(), "fingerprint-a")

        assert replay.kind is BeginKind.REPLAY
        assert replay.response == StoredResponse(status_code=504, body=body)

    asyncio.run(scenario())


def test_cached_body_is_isolated_from_caller_mutation() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        original_body = {
            "schemaVersion": 1,
            "turn": {"lines": ["해가 길을 비춰요."]},
        }
        await store.begin(_scope(), "fingerprint-a")
        await store.complete(
            _scope(),
            "fingerprint-a",
            StoredResponse(status_code=200, body=original_body),
        )

        original_body["turn"]["lines"].append("외부에서 바꾼 문장")
        first_replay = await store.begin(_scope(), "fingerprint-a")
        assert first_replay.response is not None
        first_replay.response.body["turn"]["lines"].append("재생 응답에서 바꾼 문장")

        second_replay = await store.begin(_scope(), "fingerprint-a")

        assert second_replay.response is not None
        assert second_replay.response.body == {
            "schemaVersion": 1,
            "turn": {"lines": ["해가 길을 비춰요."]},
        }

    asyncio.run(scenario())


def test_different_fingerprint_conflicts_while_in_progress_and_completed() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()

        await store.begin(_scope(), "fingerprint-a")
        pending_conflict = await store.begin(_scope(), "fingerprint-b")
        assert pending_conflict.kind is BeginKind.CONFLICT

        await store.complete(
            _scope(),
            "fingerprint-a",
            StoredResponse(status_code=200, body={"schemaVersion": 1}),
        )
        completed_conflict = await store.begin(_scope(), "fingerprint-b")
        assert completed_conflict.kind is BeginKind.CONFLICT

    asyncio.run(scenario())


def test_release_allows_retry_after_transient_failure() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        await store.begin(_scope(), "fingerprint-a")

        assert await store.release(_scope(), "fingerprint-a")
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.NEW

    asyncio.run(scenario())


def test_release_never_removes_another_or_completed_operation() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        await store.begin(_scope(), "fingerprint-a")

        assert not await store.release(_scope(), "fingerprint-b")
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.IN_PROGRESS

        await store.complete(
            _scope(),
            "fingerprint-a",
            StoredResponse(status_code=201, body={"schemaVersion": 1}),
        )
        assert not await store.release(_scope(), "fingerprint-a")
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.REPLAY

    asyncio.run(scenario())


def test_completed_response_expires_ttl_after_completion() -> None:
    async def scenario() -> None:
        now = 10.0

        def clock() -> float:
            return now

        store = MemoryIdempotencyStore(ttl_seconds=5.0, clock=clock)
        await store.begin(_scope(), "fingerprint-a")

        now = 100.0
        await store.complete(
            _scope(),
            "fingerprint-a",
            StoredResponse(status_code=200, body={"schemaVersion": 1}),
        )

        now = 104.999
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.REPLAY

        now = 105.0
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.NEW

    asyncio.run(scenario())


def test_in_progress_operation_does_not_expire_and_risk_duplicate_work() -> None:
    async def scenario() -> None:
        now = 0.0

        def clock() -> float:
            return now

        store = MemoryIdempotencyStore(ttl_seconds=1.0, clock=clock)
        await store.begin(_scope(), "fingerprint-a")

        now = 100.0
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.IN_PROGRESS

    asyncio.run(scenario())


def test_all_scope_components_partition_idempotency_keys() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        scopes = [
            _scope(),
            _scope(api_identity="other-backend"),
            _scope(method="PUT"),
            _scope(path="/api/v1/story/repairs"),
            _scope(key="a9c0f314-ca19-44ff-85e2-e99016c0d6b6"),
        ]

        results = await asyncio.gather(
            *[store.begin(scope, "same-fingerprint") for scope in scopes]
        )

        assert all(result.kind is BeginKind.NEW for result in results)

    asyncio.run(scenario())


def test_stale_completion_does_not_create_or_overwrite_an_entry() -> None:
    async def scenario() -> None:
        store = MemoryIdempotencyStore()
        response = StoredResponse(
            status_code=200,
            body={"schemaVersion": 1},
        )

        assert not await store.complete(
            _scope(),
            "fingerprint-a",
            response,
        )
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.NEW
        assert not await store.complete(
            _scope(),
            "fingerprint-b",
            response,
        )
        assert (await store.begin(_scope(), "fingerprint-a")).kind is BeginKind.IN_PROGRESS

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "ttl_seconds",
    [0.0, -1.0, math.inf, -math.inf, math.nan],
)
def test_ttl_must_be_positive_and_finite(ttl_seconds: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        MemoryIdempotencyStore(ttl_seconds=ttl_seconds)
