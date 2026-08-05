from __future__ import annotations

import re
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



_SINGLE_CURLY_PAIR_PATTERN = re.compile(r"‘([^‘’]+)’")
_CURLY_DIALOGUE_BODY_PATTERN = re.compile(r"“([^“”]+)”")


def normalize_dialogue_quotes(sentence: str) -> str:
    """생성 모델의 대화 표기를 품질 계약 형식(“…!”)으로 정규화한다.

    계약(CURLY_DIALOGUE_FORMAT)은 둥근 큰따옴표와 인용부 끝 문장 부호를
    요구하지만 모델은 곧은따옴표(")나 홑따옴표(‘’)를 자주 쓴다.
    표기 차이만으로 같은 문장이 반복 탈락해 생성 전체가 실패하지 않도록
    결정적으로 고쳐 준다. 짝이 맞지 않는 따옴표는 건드리지 않는다.
    """
    normalized = _replace_alternating_straight_quotes(sentence)
    normalized = _SINGLE_CURLY_PAIR_PATTERN.sub(r"“\1”", normalized)
    return _CURLY_DIALOGUE_BODY_PATTERN.sub(_ensure_dialogue_end_punctuation, normalized)


def _replace_alternating_straight_quotes(sentence: str) -> str:
    if sentence.count('"') % 2 != 0:
        return sentence
    parts = sentence.split('"')
    rebuilt: list[str] = [parts[0]]
    for index, part in enumerate(parts[1:]):
        rebuilt.append("“" if index % 2 == 0 else "”")
        rebuilt.append(part)
    return "".join(rebuilt)


def _ensure_dialogue_end_punctuation(match: re.Match[str]) -> str:
    body = match.group(1).rstrip()
    if body and body[-1] in ".!?":
        return f"“{body}”"
    if body.endswith(","):
        body = body[:-1]
    return f"“{body}.”"




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
    "normalize_dialogue_quotes",
    "normalize_text",
    "written_syllable_count",
]
