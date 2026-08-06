from __future__ import annotations

from dataclasses import dataclass

from iread_ai.personalization.hangul import (
    ASPIRATED_ONSETS,
    COMPLEX_CODA_PARTS,
    COMPOUND_VOWELS,
    TENSE_ONSETS,
    decompose_text,
)

BANK_VERSION = "4"


@dataclass(frozen=True, slots=True)
class UnitSeed:
    unit_type: str
    surface: str
    spoken_text: str
    pronunciation: str
    difficulty: int
    familiarity: int
    trace_asset_key: str | None = None


CONSONANTS = (
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅅ",
    "ㅇ",
    "ㅈ",
    "ㄲ",
    "ㄸ",
    "ㅃ",
    "ㅆ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)
VOWELS = (
    "ㅏ",
    "ㅑ",
    "ㅓ",
    "ㅕ",
    "ㅗ",
    "ㅛ",
    "ㅜ",
    "ㅠ",
    "ㅡ",
    "ㅣ",
    "ㅐ",
    "ㅔ",
    "ㅒ",
    "ㅖ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅢ",
)

_TRACE_ASSET_KEYS = {
    ("CONSONANT", "ㄱ"): "consonant_0",
    ("CONSONANT", "ㄴ"): "consonant_1",
    ("CONSONANT", "ㄷ"): "consonant_2",
    ("CONSONANT", "ㄹ"): "consonant_3",
    ("CONSONANT", "ㅁ"): "consonant_4",
    ("VOWEL", "ㅏ"): "vowel_0",
    ("VOWEL", "ㅓ"): "vowel_1",
    ("VOWEL", "ㅗ"): "vowel_2",
    ("VOWEL", "ㅜ"): "vowel_3",
    ("VOWEL", "ㅣ"): "vowel_4",
    ("SYLLABLE", "가"): "syllable_0",
    ("SYLLABLE", "너"): "syllable_1",
    ("SYLLABLE", "도"): "syllable_2",
    ("SYLLABLE", "모"): "syllable_3",
    ("SYLLABLE", "비"): "syllable_4",
}

_SYLLABLE_DATA = (
    ("가", "가", 1),
    ("나", "나", 1),
    ("다", "다", 1),
    ("라", "라", 1),
    ("마", "마", 1),
    ("바", "바", 1),
    ("사", "사", 1),
    ("아", "아", 1),
    ("자", "자", 1),
    ("차", "차", 2),
    ("카", "카", 2),
    ("타", "타", 2),
    ("파", "파", 2),
    ("하", "하", 2),
    ("너", "너", 1),
    ("도", "도", 1),
    ("모", "모", 1),
    ("비", "비", 1),
    ("거", "거", 1),
    ("고", "고", 1),
    ("구", "구", 1),
    ("그", "그", 1),
    ("기", "기", 1),
    ("각", "각", 2),
    ("간", "간", 2),
    ("갇", "갇", 2),
    ("갈", "갈", 2),
    ("감", "감", 2),
    ("갑", "갑", 2),
    ("갓", "갇", 3),
    ("강", "강", 2),
    ("값", "갑", 4),
    ("닭", "닥", 4),
    ("삶", "삼", 4),
)
_WORD_DATA = (
    ("가방", 2),
    ("나무", 1),
    ("다리", 1),
    ("라디오", 3),
    ("모자", 1),
    ("바다", 1),
    ("사과", 2),
    ("아기", 1),
    ("자두", 1),
    ("꼬리", 2),
    ("딸기", 2),
    ("뿌리", 2),
    ("쌀", 2),
    ("짝", 2),
    ("참새", 2),
    ("코", 2),
    ("토끼", 2),
    ("파도", 2),
    ("하마", 2),
    ("기차", 2),
    ("노래", 2),
    ("도토리", 2),
    ("머리", 1),
    ("비누", 1),
    ("수박", 2),
    ("오리", 1),
    ("주머니", 2),
    ("치마", 2),
    ("쿠키", 3),
    ("타조", 2),
    ("포도", 2),
    ("호수", 2),
    ("산", 2),
    ("달", 2),
    ("밤", 2),
    ("집", 2),
    ("공", 2),
    ("문", 2),
    ("손", 2),
    ("눈", 2),
    ("나비", 1),
    ("바지", 1),
    ("과자", 1),
    ("기린", 2),
    ("고래", 2),
    ("구두", 1),
    ("가위", 2),
    ("가구", 1),
    ("강아지", 2),
    ("고구마", 2),
    ("까치", 2),
    ("토마토", 2),
    ("무지개", 2),
    ("사슴", 2),
    ("연필", 2),
    ("책상", 2),
    ("학교", 2),
    ("친구", 1),
    ("하늘", 1),
    ("구름", 1),
    ("바람", 1),
    ("꽃", 2),
    ("물고기", 2),
    ("거북이", 2),
    ("여우", 1),
    ("곰", 1),
    ("새", 1),
    ("해", 1),
    ("별", 1),
    ("우산", 1),
    ("버스", 2),
    ("소리", 1),
    ("마음", 1),
    ("웃음", 2),
    ("노을", 2),
    ("마을", 1),
    ("시계", 2),
    ("접시", 1),
    ("우유", 1),
    ("주스", 2),
)

CONFUSION_PAIRS = (
    ("CONSONANT", "ㄱ", "ㅋ", "SOUND", 1),
    ("CONSONANT", "ㄱ", "ㄲ", "SOUND", 1),
    ("CONSONANT", "ㄷ", "ㅌ", "SOUND", 1),
    ("CONSONANT", "ㄷ", "ㄸ", "SOUND", 1),
    ("CONSONANT", "ㅂ", "ㅍ", "SOUND", 1),
    ("CONSONANT", "ㅂ", "ㅃ", "SOUND", 1),
    ("CONSONANT", "ㅅ", "ㅆ", "SOUND", 1),
    ("CONSONANT", "ㅈ", "ㅊ", "SOUND", 1),
    ("CONSONANT", "ㅈ", "ㅉ", "SOUND", 1),
    ("CONSONANT", "ㄴ", "ㄹ", "SHAPE", 2),
    ("CONSONANT", "ㅁ", "ㅂ", "SHAPE", 2),
    ("VOWEL", "ㅏ", "ㅓ", "SOUND", 1),
    ("VOWEL", "ㅑ", "ㅕ", "SOUND", 1),
    ("VOWEL", "ㅗ", "ㅜ", "SHAPE", 1),
    ("VOWEL", "ㅛ", "ㅠ", "SHAPE", 1),
    ("VOWEL", "ㅐ", "ㅔ", "SOUND", 1),
    ("VOWEL", "ㅙ", "ㅞ", "SOUND", 1),
    ("VOWEL", "ㅘ", "ㅝ", "SHAPE", 2),
    ("VOWEL", "ㅚ", "ㅟ", "SOUND", 2),
)


def unit_seeds() -> tuple[UnitSeed, ...]:
    consonants = tuple(
        UnitSeed(
            "CONSONANT",
            value,
            value,
            value,
            2 if value in TENSE_ONSETS or value in ASPIRATED_ONSETS or value == "ㅎ" else 1,
            5 if value in {"ㄱ", "ㄴ", "ㄷ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ"} else 4,
            _TRACE_ASSET_KEYS.get(("CONSONANT", value)),
        )
        for value in CONSONANTS
    )
    vowels = tuple(
        UnitSeed(
            "VOWEL",
            value,
            value,
            value,
            3 if value in COMPOUND_VOWELS else 2 if value in {"ㅐ", "ㅔ", "ㅒ", "ㅖ"} else 1,
            5 if value in {"ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"} else 4,
            _TRACE_ASSET_KEYS.get(("VOWEL", value)),
        )
        for value in VOWELS
    )
    syllables = tuple(
        UnitSeed(
            "SYLLABLE",
            surface,
            surface,
            pronunciation,
            difficulty,
            5,
            _TRACE_ASSET_KEYS.get(("SYLLABLE", surface)),
        )
        for surface, pronunciation, difficulty in _SYLLABLE_DATA
    )
    words = tuple(
        UnitSeed("WORD", surface, surface, surface, difficulty, 5)
        for surface, difficulty in _WORD_DATA
    )
    return consonants + vowels + syllables + words


def unit_parts(seed: UnitSeed) -> tuple[str | None, str | None, str | None]:
    if seed.unit_type == "CONSONANT":
        return seed.surface, None, None
    if seed.unit_type == "VOWEL":
        return None, seed.surface, None
    syllables = decompose_text(seed.surface)
    if not syllables:
        return None, None, None
    return syllables[0].onset, syllables[0].nucleus, syllables[-1].coda or None


def unit_features(seed: UnitSeed) -> frozenset[str]:
    features: set[str] = set()
    if seed.unit_type == "CONSONANT":
        kind = (
            "TENSE"
            if seed.surface in TENSE_ONSETS
            else "ASPIRATED"
            if seed.surface in ASPIRATED_ONSETS or seed.surface == "ㅎ"
            else "BASIC"
        )
        features.add(f"GRAPHEME.ONSET.{kind}.{seed.surface}")
        if seed.surface not in {"ㄸ", "ㅃ", "ㅉ"}:
            features.add(f"GRAPHEME.CODA.SIMPLE.{seed.surface}")
        return frozenset(features)
    if seed.unit_type == "VOWEL":
        kind = "COMPOUND" if seed.surface in COMPOUND_VOWELS else "BASIC"
        return frozenset({f"GRAPHEME.VOWEL.{kind}.{seed.surface}"})
    syllables = decompose_text(seed.surface)
    for syllable in syllables:
        onset_kind = (
            "TENSE"
            if syllable.onset in TENSE_ONSETS
            else "ASPIRATED"
            if syllable.onset in ASPIRATED_ONSETS or syllable.onset == "ㅎ"
            else "BASIC"
        )
        vowel_kind = "COMPOUND" if syllable.nucleus in COMPOUND_VOWELS else "BASIC"
        features.add(f"GRAPHEME.ONSET.{onset_kind}.{syllable.onset}")
        features.add(f"GRAPHEME.VOWEL.{vowel_kind}.{syllable.nucleus}")
        if syllable.coda:
            coda_kind = "COMPLEX" if syllable.coda in COMPLEX_CODA_PARTS else "SIMPLE"
            features.add(f"GRAPHEME.CODA.{coda_kind}.{syllable.coda}")
    if seed.unit_type == "SYLLABLE" and syllables:
        features.add("SYLLABLE.CVC" if syllables[0].coda else "SYLLABLE.CV")
        if syllables[0].onset in TENSE_ONSETS:
            features.add("SYLLABLE.TENSE_ONSET")
        if syllables[0].coda in COMPLEX_CODA_PARTS:
            features.add("SYLLABLE.COMPLEX_CODA")
    if seed.unit_type == "WORD":
        features.add(f"WORD.SYLLABLE_COUNT.{min(len(syllables), 5)}")
    return frozenset(features)


__all__ = [
    "BANK_VERSION",
    "CONFUSION_PAIRS",
    "UnitSeed",
    "unit_features",
    "unit_parts",
    "unit_seeds",
]
