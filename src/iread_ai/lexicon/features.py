from __future__ import annotations

from collections.abc import Iterable

_BASIC_ONSETS = frozenset("ㄱㄴㄷㄹㅁㅂㅅㅇㅈ")
_TENSE_ONSETS = frozenset("ㄲㄸㅃㅆㅉ")
_ASPIRATED_ONSETS = frozenset("ㅊㅋㅌㅍㅎ")
_COMPOUND_VOWELS = frozenset("ㅘㅙㅚㅝㅞㅟㅢ")
_COMPLEX_CODAS = frozenset("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")
_PHONOLOGY_ALIASES = {
    "PHONOLOGY.LIAISON": "PHONO_LIAISON",
    "PHONOLOGY.NASALIZATION": "PHONO_NASALIZATION",
    "PHONOLOGY.LIQUIDIZATION": "PHONO_LIQUIDIZATION",
    "PHONOLOGY.PALATALIZATION": "PHONO_PALATALIZATION",
    "PHONOLOGY.TENSIFICATION": "PHONO_TENSIFICATION",
    "PHONOLOGY.ASPIRATION": "PHONO_ASPIRATION",
}


def canonical_feature_code(code: str) -> str:
    normalized = code.strip()
    normalized = _PHONOLOGY_ALIASES.get(normalized, normalized)
    if normalized == "SYLLABLE.COMPLEX_CODA":
        return "HAS_COMPLEX_CODA"
    if normalized.startswith("GRAPHEME.") or normalized.startswith("PHONO_"):
        return normalized
    if normalized.startswith("ONSET_"):
        value = normalized.removeprefix("ONSET_")
        if value in _TENSE_ONSETS:
            family = "TENSE"
        elif value in _ASPIRATED_ONSETS:
            family = "ASPIRATED"
        else:
            family = "BASIC"
        return f"GRAPHEME.ONSET.{family}.{value}"
    if normalized.startswith("NUCLEUS_"):
        value = normalized.removeprefix("NUCLEUS_")
        family = "COMPOUND" if value in _COMPOUND_VOWELS else "BASIC"
        return f"GRAPHEME.VOWEL.{family}.{value}"
    if normalized.startswith("CODA_"):
        value = normalized.removeprefix("CODA_")
        family = "COMPLEX" if value in _COMPLEX_CODAS else "SIMPLE"
        return f"GRAPHEME.CODA.{family}.{value}"
    return normalized


def storage_feature_codes(code: str) -> tuple[str, ...]:
    normalized = canonical_feature_code(code)
    if normalized == "GRAPHEME.ONSET.TENSE":
        return ("HAS_TENSE_ONSET",)
    if normalized == "GRAPHEME.ONSET.ASPIRATED":
        return ("HAS_ASPIRATED_ONSET",)
    if normalized == "GRAPHEME.VOWEL.COMPOUND":
        return ("HAS_COMPOUND_VOWEL",)
    if normalized == "GRAPHEME.CODA.COMPLEX":
        return ("HAS_COMPLEX_CODA",)
    if normalized.startswith("GRAPHEME.ONSET."):
        return (f"ONSET_{normalized.rsplit('.', 1)[-1]}",)
    if normalized.startswith("GRAPHEME.VOWEL."):
        return (f"NUCLEUS_{normalized.rsplit('.', 1)[-1]}",)
    if normalized.startswith("GRAPHEME.CODA."):
        return (f"CODA_{normalized.rsplit('.', 1)[-1]}",)
    return (normalized,)


def canonicalize_features(features: Iterable[tuple[str, int]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for code, count in features:
        canonical = canonical_feature_code(code)
        result[canonical] = result.get(canonical, 0) + int(count)
    return result


def feature_matches(actual: str, requested: str) -> bool:
    return actual == requested or actual.startswith(f"{requested}.")


__all__ = [
    "canonical_feature_code",
    "canonicalize_features",
    "feature_matches",
    "storage_feature_codes",
]
