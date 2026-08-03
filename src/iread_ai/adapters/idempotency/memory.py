from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from iread_ai.ports.idempotency_store import (
    BeginKind,
    BeginResult,
    IdempotencyScope,
    IdempotencyStore,
    StoredResponse,
)


@dataclass(slots=True)
class _Entry:
    fingerprint: str
    response: StoredResponse | None = None
    expires_at: float | None = None

    @property
    def in_progress(self) -> bool:
        return self.response is None


class MemoryIdempotencyStore(IdempotencyStore):
    def __init__(
        self,
        *,
        ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive finite number.")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[IdempotencyScope, _Entry] = {}
        self._lock = asyncio.Lock()

    async def begin(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
    ) -> BeginResult:
        async with self._lock:
            now = self._clock()
            self._discard_expired(now)
            entry = self._entries.get(scope)

            if entry is None:
                self._entries[scope] = _Entry(fingerprint=fingerprint)
                return BeginResult(BeginKind.NEW)

            if entry.fingerprint != fingerprint:
                return BeginResult(BeginKind.CONFLICT)

            if entry.in_progress:
                return BeginResult(BeginKind.IN_PROGRESS)

            assert entry.response is not None
            return BeginResult(
                BeginKind.REPLAY,
                response=self._copy_response(entry.response),
            )

    async def complete(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
        response: StoredResponse,
    ) -> bool:
        async with self._lock:
            now = self._clock()
            self._discard_expired(now)
            entry = self._entries.get(scope)
            if entry is None or entry.fingerprint != fingerprint or not entry.in_progress:
                return False

            entry.response = self._copy_response(response)
            entry.expires_at = now + self._ttl_seconds
            return True

    async def release(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
    ) -> bool:
        async with self._lock:
            self._discard_expired(self._clock())
            entry = self._entries.get(scope)
            if entry is None or entry.fingerprint != fingerprint or not entry.in_progress:
                return False

            del self._entries[scope]
            return True

    def _discard_expired(self, now: float) -> None:
        expired = [
            scope
            for scope, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for scope in expired:
            del self._entries[scope]

    @staticmethod
    def _copy_response(response: StoredResponse) -> StoredResponse:
        return StoredResponse(
            status_code=response.status_code,
            body=deepcopy(response.body),
        )
