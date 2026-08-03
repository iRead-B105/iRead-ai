"""Small in-memory fingerprint idempotency store for a single AI process."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any


class IdempotencyConflict(RuntimeError):
    pass


class IdempotencyInProgress(RuntimeError):
    pass


@dataclass(slots=True)
class _Entry:
    fingerprint: str
    value: Any = None
    completed: bool = False
    expires_at: float | None = None


class MemoryIdempotencyStore:
    def __init__(
        self,
        *,
        ttl_seconds: float = 600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("idempotency TTL must be a positive finite value")
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = Lock()

    def execute(
        self,
        *,
        scope: str,
        key: str,
        payload: Any,
        action: Callable[[], Any],
    ) -> tuple[Any, bool]:
        fingerprint = request_fingerprint(payload)
        entry_key = (scope, key)
        with self._lock:
            self._discard_expired()
            entry = self._entries.get(entry_key)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise IdempotencyConflict(
                        "the idempotency key was already used with another request"
                    )
                if not entry.completed:
                    raise IdempotencyInProgress(
                        "the request with this idempotency key is still running"
                    )
                return deepcopy(entry.value), True
            self._entries[entry_key] = _Entry(fingerprint=fingerprint)

        try:
            value = action()
        except Exception:
            with self._lock:
                current = self._entries.get(entry_key)
                if current is not None and current.fingerprint == fingerprint:
                    del self._entries[entry_key]
            raise

        with self._lock:
            entry = self._entries[entry_key]
            entry.value = deepcopy(value)
            entry.completed = True
            entry.expires_at = self._clock() + self._ttl_seconds
        return value, False

    def _discard_expired(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.expires_at is not None and entry.expires_at <= now
        ]
        for key in expired:
            del self._entries[key]


def request_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
