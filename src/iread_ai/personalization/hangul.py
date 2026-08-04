from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass

ONSETS = (
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

NUCLEI = (
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
)

CODAS = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

COMPLEX_CODA_PARTS = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}

TENSE_ONSETS = frozenset({"ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"})
ASPIRATED_ONSETS = frozenset({"ㅋ", "ㅌ", "ㅍ", "ㅊ"})
COMPOUND_VOWELS = frozenset({"ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"})
GLIDE_VOWELS = frozenset({"ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ"})
DOUBLE_CODAS = frozenset({"ㄲ", "ㅆ"})


@dataclass(frozen=True, slots=True)
class HangulSyllable:
    char: str
    onset: str
    nucleus: str
    coda: str
    coda_parts: tuple[str, ...]
    source_position: int


def normalize_text(text: str | None) -> str:
    return unicodedata.normalize("NFC", text or "")


def decompose_syllable(char: str, source_position: int = 0) -> HangulSyllable | None:
    if len(char) != 1 or not ("\uac00" <= char <= "\ud7a3"):
        return None
    offset = ord(char) - 0xAC00
    onset = ONSETS[offset // 588]
    nucleus = NUCLEI[(offset % 588) // 28]
    coda = CODAS[offset % 28]
    return HangulSyllable(
        char=char,
        onset=onset,
        nucleus=nucleus,
        coda=coda,
        coda_parts=COMPLEX_CODA_PARTS.get(coda, (coda,) if coda else ()),
        source_position=source_position,
    )


def decompose_text(text: str | None) -> tuple[HangulSyllable, ...]:
    return tuple(
        syllable
        for index, char in enumerate(normalize_text(text))
        if (syllable := decompose_syllable(char, index)) is not None
    )


def mask_protected_terms(text: str, protected_terms: tuple[str, ...]) -> str:
    masked = normalize_text(text)
    terms = sorted(
        {normalize_text(term).strip() for term in protected_terms if normalize_text(term).strip()},
        key=len,
        reverse=True,
    )
    for term in terms:
        masked = masked.replace(term, " " * len(term))
    return masked


def count_surface_features(text: str | None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for syllable in decompose_text(text):
        counts[f"ONSET_{syllable.onset}"] += 1
        counts[f"NUCLEUS_{syllable.nucleus}"] += 1
        if syllable.coda:
            counts[f"CODA_{syllable.coda}"] += 1
            counts["HAS_BATCHIM"] += 1
        if syllable.coda in COMPLEX_CODA_PARTS:
            counts["HAS_COMPLEX_CODA"] += 1
        if syllable.coda in DOUBLE_CODAS:
            counts["HAS_DOUBLE_CODA"] += 1
        if syllable.onset in TENSE_ONSETS:
            counts["HAS_TENSE_ONSET"] += 1
        if syllable.onset in ASPIRATED_ONSETS:
            counts["HAS_ASPIRATED_ONSET"] += 1
        if syllable.nucleus in COMPOUND_VOWELS:
            counts["HAS_COMPOUND_VOWEL"] += 1
        if syllable.nucleus in GLIDE_VOWELS:
            counts["HAS_GLIDE_VOWEL"] += 1
    return dict(sorted(counts.items()))


def written_syllable_count(text: str | None) -> int:
    return len(decompose_text(text))


__all__ = [
    "ASPIRATED_ONSETS",
    "CODAS",
    "COMPLEX_CODA_PARTS",
    "COMPOUND_VOWELS",
    "DOUBLE_CODAS",
    "GLIDE_VOWELS",
    "HangulSyllable",
    "NUCLEI",
    "ONSETS",
    "TENSE_ONSETS",
    "count_surface_features",
    "decompose_syllable",
    "decompose_text",
    "mask_protected_terms",
    "normalize_text",
    "written_syllable_count",
]
