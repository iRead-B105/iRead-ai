from __future__ import annotations

from collections.abc import Iterable

_SUPPORTED_FAMILIES: dict[str, frozenset[str]] = {
    "VOWEL_TRACE": frozenset({"VOWEL"}),
    "CONSONANT_TRACE": frozenset({"ONSET"}),
    "SYLLABLE_TRACE": frozenset({"ONSET", "VOWEL", "CODA", "SYLLABLE"}),
    "CONSONANT_SOUND_CHOICE": frozenset({"ONSET"}),
    "VOWEL_SOUND_CHOICE": frozenset({"VOWEL"}),
    "CONSONANT_VOWEL_CLASSIFICATION": frozenset({"ONSET", "VOWEL"}),
    "SYLLABLE_INITIAL_CHOICE": frozenset({"ONSET"}),
    "WORD_INITIAL_CHOICE": frozenset({"ONSET"}),
    "SAME_INITIAL_WORD_CHOICE": frozenset({"ONSET"}),
    "FINAL_CONSONANT_CHOICE": frozenset({"CODA"}),
    "WORD_FINAL_SOUND_CHOICE": frozenset({"CODA"}),
    "FINAL_CONSONANT_COMPARISON": frozenset({"CODA"}),
    "SIMILAR_SOUND_CHOICE": frozenset({"ONSET"}),
    "PHONEME_BLEND": frozenset({"ONSET", "VOWEL", "CODA", "SYLLABLE"}),
    "SYLLABLE_BLEND": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
    "BASIC_SYLLABLE_BUILD": frozenset({"ONSET", "VOWEL", "SYLLABLE"}),
    "FINAL_SYLLABLE_BUILD": frozenset({"ONSET", "VOWEL", "CODA", "SYLLABLE"}),
    "DOUBLE_FINAL_BUILD": frozenset({"CODA", "SYLLABLE"}),
    "FINAL_CONSONANT_DELETE": frozenset({"CODA", "SYLLABLE"}),
    "SYLLABLE_DELETE": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
    "SYLLABLE_REPLACE": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
    "WORD_READING": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
    "NONWORD_READING": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
    "WORD_CHAIN_READING": frozenset(
        {"ONSET", "VOWEL", "CODA", "SYLLABLE", "WORD", "PHONOLOGY"}
    ),
}


def feature_family(feature_code: str) -> str:
    if feature_code.startswith("GRAPHEME.VOWEL."):
        return "VOWEL"
    if feature_code.startswith("GRAPHEME.ONSET."):
        return "ONSET"
    if feature_code.startswith("GRAPHEME.CODA."):
        return "CODA"
    if feature_code.startswith("SYLLABLE."):
        return "SYLLABLE"
    if feature_code.startswith("WORD."):
        return "WORD"
    if feature_code.startswith(("PHONOLOGY.", "PHONO_")):
        return "PHONOLOGY"
    if feature_code.startswith("SENTENCE."):
        return "SENTENCE"
    return "OTHER"


def feature_is_compatible(training_type: str, feature_code: str) -> bool:
    supported = _SUPPORTED_FAMILIES.get(training_type)
    if supported is None:
        return True
    return feature_family(feature_code) in supported


def compatible_features[Feature](
    training_type: str,
    features: Iterable[Feature],
) -> list[Feature]:
    return [
        feature
        for feature in features
        if feature_is_compatible(training_type, str(getattr(feature, "featureCode", feature)))
    ]


__all__ = ["compatible_features", "feature_family", "feature_is_compatible"]
