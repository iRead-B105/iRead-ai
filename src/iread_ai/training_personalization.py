from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .generation_models import (
    TrainingCandidateFit,
    TrainingPersonalizationEvidence,
)
from .lexicon.features import canonical_feature_code, canonicalize_features, feature_matches
from .personalization.analyzer import KoreanReadingAnalyzer
from .personalization.hangul import count_surface_features, decompose_text
from .training_length_policy import (
    length_evaluation,
    reading_sentences,
    training_length_policy,
)


def select_training_candidate(
    candidates: list[dict[str, Any]],
    *,
    target_features: Iterable[str],
    excluded_features: Iterable[str],
    recommended_words: Iterable[str],
    analyzer: KoreanReadingAnalyzer | None,
    lexicon_applied: bool,
    training_type: str,
    difficulty: int,
) -> tuple[dict[str, Any], TrainingPersonalizationEvidence]:
    targets = tuple(canonical_feature_code(code) for code in target_features)
    excluded = tuple(canonical_feature_code(code) for code in excluded_features)
    palette = tuple(dict.fromkeys(word for word in recommended_words if word))
    fits = [
        _evaluate_candidate(
            candidate,
            index=index,
            targets=targets,
            excluded=excluded,
            palette=palette,
            analyzer=analyzer,
            lexicon_applied=lexicon_applied,
            training_type=training_type,
            difficulty=difficulty,
        )
        for index, candidate in enumerate(candidates)
    ]
    selected = max(fits, key=candidate_fit_rank)
    return dict(candidates[selected.candidateIndex]), TrainingPersonalizationEvidence(
        lexiconApplied=lexicon_applied,
        recommendedWords=list(palette),
        selectedCandidateIndex=selected.candidateIndex,
        candidates=fits,
    )


def candidate_fit_rank(fit: TrainingCandidateFit) -> tuple[float, ...]:
    excluded_total = sum(fit.excludedOccurrences.values())
    length_rank = {
        "PASS": 2,
        "NOT_APPLICABLE": 1,
        "TOO_SHORT": 0,
        "TOO_LONG": 0,
    }.get(fit.lengthStatus, 0)
    target_rank = {
        "PASS": 2,
        "NOT_APPLICABLE": 1,
        "TOO_FEW": 0,
        "EXCESSIVE": 0,
    }.get(fit.targetLoadStatus, 0)
    return (
        float(-excluded_total),
        float(length_rank),
        float(target_rank),
        fit.score,
        fit.lengthAdjustment,
        float(-fit.candidateIndex),
    )


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    index: int,
    targets: tuple[str, ...],
    excluded: tuple[str, ...],
    palette: tuple[str, ...],
    analyzer: KoreanReadingAnalyzer | None,
    lexicon_applied: bool,
    training_type: str,
    difficulty: int,
) -> TrainingCandidateFit:
    reading_parts = reading_sentences(candidate, training_type)
    text_parts = reading_parts or tuple(_iter_text(candidate))
    joined = " ".join(text_parts)
    analysis_error: str | None = None
    if analyzer is None:
        features = canonicalize_features(count_surface_features(joined).items())
        phonology: dict[str, int] = {}
        analysis_status = "SURFACE_ONLY"
    else:
        result = analyzer.analyze(text_parts)
        features = canonicalize_features(result.surface_feature_counts.items())
        phonology = {
            canonical_feature_code(code): count
            for code, count in result.phonological_rule_counts.items()
        }
        analysis_status = result.status.value
        analysis_error = result.error
    combined = dict(features)
    for code, count in phonology.items():
        combined[code] = combined.get(code, 0) + count

    target_counts = {
        target: sum(count for actual, count in combined.items() if feature_matches(actual, target))
        for target in targets
    }
    structured_counts = {
        target: _structured_target_count(candidate, training_type, target)
        for target in targets
    }
    if targets and all(count is not None for count in structured_counts.values()):
        target_counts = {
            target: int(structured_counts[target] or 0)
            for target in targets
        }
        analysis_status = "STRUCTURED_VERIFIED"
    excluded_counts = {
        feature: sum(
            count for actual, count in combined.items() if feature_matches(actual, feature)
        )
        for feature in excluded
    }
    used_words = [word for word in palette if _palette_word_used(word, joined)]
    sentence_counts, total_syllables, length_status, length_adjustment = length_evaluation(
        reading_parts,
        training_length_policy(training_type, difficulty),
    )

    score = 100.0
    target_bounds = {
        target: _target_load_bounds(candidate, training_type, target, len(targets))
        for target in targets
    }
    target_total = sum(target_counts.values())
    if not targets:
        target_load_status = "NOT_APPLICABLE"
    elif any(
        target_counts[target] < target_bounds[target][0]
        for target in targets
    ):
        target_load_status = "TOO_FEW"
    elif any(
        target_counts[target] > target_bounds[target][1]
        for target in targets
    ):
        target_load_status = "EXCESSIVE"
    else:
        target_load_status = "PASS"
    for target, count in target_counts.items():
        minimum, maximum = target_bounds[target]
        score += min(count, maximum) * 15
        score -= max(minimum - count, 0) * 40
        score -= max(count - maximum, 0) * 18
    score -= sum(excluded_counts.values()) * 60
    score += length_adjustment
    if lexicon_applied:
        score += min(len(used_words), 3) * 8
        if not used_words:
            score -= 10
        score -= max(len(used_words) - 4, 0) * 12
    if analysis_status == "UNRELIABLE":
        score -= 5
    elif analysis_status == "SURFACE_ONLY":
        score -= 10
    return TrainingCandidateFit(
        candidateIndex=index,
        score=round(score, 3),
        targetOccurrences=target_counts,
        excludedOccurrences=excluded_counts,
        paletteWordUses=used_words,
        analysisStatus=analysis_status,
        analysisError=analysis_error,
        writtenSyllableCount=total_syllables,
        sentenceSyllableCounts=sentence_counts,
        lengthStatus=length_status,
        lengthAdjustment=length_adjustment,
        targetLoadStatus=target_load_status,
        targetOccurrenceTotal=target_total,
    )


def _target_load_bounds(
    candidate: dict[str, Any],
    training_type: str,
    target: str,
    target_count: int,
) -> tuple[int, int]:
    if training_type == "SHORT_STORY_READING":
        return 2, 6
    if training_type in {"WORD_READING", "WORD_CHAIN_READING"}:
        unit_count = max(len(_structured_target_units(candidate, training_type)), 1)
        if target.startswith("WORD.SYLLABLE_COUNT."):
            return unit_count, unit_count
        if target_count > 1:
            balanced_count = min(2, unit_count)
            return balanced_count, balanced_count
        return 1, unit_count
    return 1, 2


def _palette_word_used(word: str, text: str) -> bool:
    if re.search(rf"(?<![가-힣]){re.escape(word)}", text):
        return True
    stem = word[:-1]
    return (
        word.endswith("다")
        and sum("가" <= character <= "힣" for character in stem) >= 2
        and re.search(rf"(?<![가-힣]){re.escape(stem)}", text) is not None
    )


_STRUCTURED_TARGET_TYPES = frozenset(
    {
        "VOWEL_TRACE",
        "CONSONANT_TRACE",
        "SYLLABLE_TRACE",
        "CONSONANT_SOUND_CHOICE",
        "VOWEL_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "SYLLABLE_INITIAL_CHOICE",
        "WORD_INITIAL_CHOICE",
        "SAME_INITIAL_WORD_CHOICE",
        "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
        "FINAL_CONSONANT_COMPARISON",
        "SIMILAR_SOUND_CHOICE",
        "PHONEME_BLEND",
        "SYLLABLE_BLEND",
        "BASIC_SYLLABLE_BUILD",
        "FINAL_SYLLABLE_BUILD",
        "DOUBLE_FINAL_BUILD",
        "FINAL_CONSONANT_DELETE",
        "SYLLABLE_DELETE",
        "SYLLABLE_REPLACE",
        "WORD_READING",
        "NONWORD_READING",
        "WORD_CHAIN_READING",
    }
)
_CONSONANT_NAMES = {
    "ㄱ": "기역",
    "ㄲ": "쌍기역",
    "ㄴ": "니은",
    "ㄷ": "디귿",
    "ㄸ": "쌍디귿",
    "ㄹ": "리을",
    "ㅁ": "미음",
    "ㅂ": "비읍",
    "ㅃ": "쌍비읍",
    "ㅅ": "시옷",
    "ㅆ": "쌍시옷",
    "ㅇ": "이응",
    "ㅈ": "지읒",
    "ㅉ": "쌍지읒",
    "ㅊ": "치읓",
    "ㅋ": "키읔",
    "ㅌ": "티읕",
    "ㅍ": "피읖",
    "ㅎ": "히읗",
}


def _structured_target_count(
    candidate: dict[str, Any],
    training_type: str,
    target: str,
) -> int | None:
    if training_type not in _STRUCTURED_TARGET_TYPES:
        return None
    units = _structured_target_units(candidate, training_type)
    if training_type == "SYLLABLE_DELETE" and target.startswith("WORD.SYLLABLE_COUNT."):
        expected = int(target.rsplit(".", 1)[-1])
        source_count = len(decompose_text(str(candidate.get("source", ""))))
        result_count = len(decompose_text(str(candidate.get("result", ""))))
        return int(expected in {source_count, result_count})
    if target.startswith("GRAPHEME.ONSET."):
        if target == "GRAPHEME.ONSET.TENSE":
            return sum(
                count_surface_features(unit).get("HAS_TENSE_ONSET", 0)
                for unit in units
            )
        value = target.rsplit(".", 1)[-1]
        return sum(
            sum(syllable.onset == value for syllable in decompose_text(unit))
            + unit.count(value)
            + int(_CONSONANT_NAMES.get(value, "") in unit)
            for unit in units
        )
    if target.startswith("GRAPHEME.VOWEL."):
        if target == "GRAPHEME.VOWEL.COMPOUND":
            return sum(
                count_surface_features(unit).get("HAS_COMPOUND_VOWEL", 0)
                for unit in units
            )
        value = target.rsplit(".", 1)[-1]
        return sum(
            sum(syllable.nucleus == value for syllable in decompose_text(unit))
            + unit.count(value)
            for unit in units
        )
    if target.startswith("GRAPHEME.CODA."):
        value = target.rsplit(".", 1)[-1]
        return sum(
            sum(syllable.coda == value for syllable in decompose_text(unit))
            for unit in units
        )
    if target == "SYLLABLE.CV":
        return sum(
            1
            for unit in units
            if len(syllables := decompose_text(unit)) == 1 and not syllables[0].coda
        )
    if target == "SYLLABLE.CVC":
        return sum(
            1
            for unit in units
            if len(syllables := decompose_text(unit)) == 1 and bool(syllables[0].coda)
        )
    if target.startswith("WORD.SYLLABLE_COUNT."):
        expected = int(target.rsplit(".", 1)[-1])
        return sum(len(decompose_text(unit)) == expected for unit in units)
    if target == "HAS_COMPLEX_CODA":
        return sum(count_surface_features(unit).get("HAS_COMPLEX_CODA", 0) for unit in units)
    return None


def _structured_target_units(
    candidate: dict[str, Any],
    training_type: str,
) -> tuple[str, ...]:
    if training_type in {"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"}:
        return (str(candidate.get("target", "")),)
    if training_type in {
        "CONSONANT_SOUND_CHOICE",
        "VOWEL_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "SYLLABLE_INITIAL_CHOICE",
        "WORD_INITIAL_CHOICE",
        "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
        "FINAL_CONSONANT_COMPARISON",
        "SIMILAR_SOUND_CHOICE",
    }:
        return (str(candidate.get("audioText", "")),)
    if training_type == "SAME_INITIAL_WORD_CHOICE":
        return (str(candidate.get("targetAudioText", "")),)
    if training_type in {
        "PHONEME_BLEND",
        "SYLLABLE_BLEND",
        "BASIC_SYLLABLE_BUILD",
        "FINAL_SYLLABLE_BUILD",
        "DOUBLE_FINAL_BUILD",
    }:
        return (str(candidate.get("result", "")),)
    if training_type in {"FINAL_CONSONANT_DELETE", "SYLLABLE_DELETE", "SYLLABLE_REPLACE"}:
        return (str(candidate.get("source", "")),)
    if training_type == "WORD_READING":
        return tuple(str(value) for value in candidate.get("words", []))
    if training_type == "NONWORD_READING":
        return tuple(str(value.get("text", "")) for value in candidate.get("words", []))
    if training_type == "WORD_CHAIN_READING":
        return tuple(str(value) for value in candidate.get("words", []))
    return ()


def _iter_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {"imagePrompt", "emotion"}:
                yield from _iter_text(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_text(item)


__all__ = ["candidate_fit_rank", "select_training_candidate"]
