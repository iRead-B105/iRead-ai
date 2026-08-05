from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearningUnit:
    id: int
    unit_type: str
    surface: str
    spoken_text: str
    pronunciation: str
    onset: str | None
    vowel: str | None
    coda: str | None
    difficulty: int
    familiarity: int
    trace_asset_key: str | None
    feature_codes: frozenset[str]
    confusion_ids: tuple[int, ...]

    def matches_feature(self, requested: str) -> bool:
        return any(
            actual == requested
            or actual.startswith(f"{requested}.")
            or requested.startswith(f"{actual}.")
            for actual in self.feature_codes
        )


__all__ = ["LearningUnit"]
