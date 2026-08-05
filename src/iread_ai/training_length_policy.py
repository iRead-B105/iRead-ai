from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .personalization.hangul import written_syllable_count


@dataclass(frozen=True, slots=True)
class TrainingLengthPolicy:
    sentence_min: int
    sentence_max: int
    total_min: int | None = None
    total_max: int | None = None


_SENTENCE_RANGES = {
    1: (5, 16),
    2: (6, 15),
    3: (8, 20),
    4: (9, 24),
    5: (9, 28),
}

_PASSAGE_SENTENCE_RANGES = {
    1: (4, 12),
    2: (5, 15),
    3: (7, 20),
    4: (8, 23),
    5: (9, 26),
}

_STORY_SENTENCE_RANGES = {
    1: (3, 16),
    2: (4, 18),
    3: (4, 22),
    4: (4, 24),
    5: (4, 28),
}

_SINGLE_SENTENCE_TYPES = frozenset(
    {
        "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
    }
)

_STORY_TOTAL_RANGES = {
    1: (20, 48),
    2: (24, 56),
    3: (30, 64),
    4: (34, 72),
    5: (36, 80),
}

_PASSAGE_TOTAL_RANGES = {
    1: (10, 24),
    2: (15, 34),
    3: (20, 46),
    4: (25, 58),
    5: (30, 72),
}


def training_length_policy(
    training_type: str,
    difficulty: int,
) -> TrainingLengthPolicy | None:
    sentence_min, sentence_max = _SENTENCE_RANGES[difficulty]
    if training_type in _SINGLE_SENTENCE_TYPES:
        return TrainingLengthPolicy(sentence_min, sentence_max)
    if training_type == "SHORT_STORY_READING":
        sentence_min, sentence_max = _STORY_SENTENCE_RANGES[difficulty]
        total_min, total_max = _STORY_TOTAL_RANGES[difficulty]
        return TrainingLengthPolicy(sentence_min, sentence_max, total_min, total_max)
    if training_type == "SHORT_PASSAGE_READING":
        sentence_min, sentence_max = _PASSAGE_SENTENCE_RANGES[difficulty]
        total_min, total_max = _PASSAGE_TOTAL_RANGES[difficulty]
        return TrainingLengthPolicy(sentence_min, sentence_max, total_min, total_max)
    return None


def reading_sentences(candidate: dict[str, Any], training_type: str) -> tuple[str, ...]:
    if training_type == "SHORT_STORY_READING":
        values = candidate.get("sentences", [])
        return tuple(
            str(value.get("text", "")).strip()
            for value in values
            if isinstance(value, dict) and str(value.get("text", "")).strip()
        )
    if training_type == "SHORT_PASSAGE_READING":
        values = candidate.get("sentences", [])
        return tuple(str(value).strip() for value in values if str(value).strip())
    if training_type == "FILL_IN_THE_BLANK":
        value = candidate.get("completedSentence") or candidate.get("sentence")
    else:
        value = candidate.get("sentence") or candidate.get("completedSentence")
    return (str(value).strip(),) if value and str(value).strip() else ()


def length_evaluation(
    sentences: tuple[str, ...],
    policy: TrainingLengthPolicy | None,
) -> tuple[list[int], int, str, float]:
    counts = [written_syllable_count(sentence) for sentence in sentences]
    total = sum(counts)
    if policy is None or not counts:
        return counts, total, "NOT_APPLICABLE", 0.0

    shortfall = sum(max(policy.sentence_min - count, 0) for count in counts)
    overage = sum(max(count - policy.sentence_max, 0) for count in counts)
    total_shortfall = (
        max(policy.total_min - total, 0) if policy.total_min is not None else 0
    )
    total_overage = max(total - policy.total_max, 0) if policy.total_max is not None else 0
    if shortfall == overage == total_shortfall == total_overage == 0:
        return counts, total, "PASS", 20.0
    adjustment = -(
        shortfall * 3 + overage * 8 + total_shortfall * 2 + total_overage * 4
    )
    return counts, total, "TOO_LONG" if overage or total_overage else "TOO_SHORT", float(
        adjustment
    )


__all__ = [
    "TrainingLengthPolicy",
    "length_evaluation",
    "reading_sentences",
    "training_length_policy",
]
