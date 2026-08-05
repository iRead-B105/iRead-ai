from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class IdempotencyScope:
    api_identity: str
    method: str
    path: str
    key: str


@dataclass(frozen=True, slots=True)
class StoredResponse:
    status_code: int
    body: JsonObject


class BeginKind(StrEnum):
    NEW = "new"
    REPLAY = "replay"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True, slots=True)
class BeginResult:
    kind: BeginKind
    response: StoredResponse | None = None

    def __post_init__(self) -> None:
        if self.kind is BeginKind.REPLAY and self.response is None:
            raise ValueError("A replay result requires a stored response.")
        if self.kind is not BeginKind.REPLAY and self.response is not None:
            raise ValueError("Only a replay result may include a stored response.")


class IdempotencyStore(Protocol):
    async def begin(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
    ) -> BeginResult: ...

    async def complete(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
        response: StoredResponse,
    ) -> bool: ...

    async def release(
        self,
        scope: IdempotencyScope,
        fingerprint: str,
    ) -> bool: ...
