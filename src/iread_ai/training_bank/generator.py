from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

from iread_ai.generation_models import TrainingCandidateRequest, TrainingCandidateResponse
from iread_ai.personalization.hangul import COMPLEX_CODA_PARTS

from .models import LearningUnit
from .repository import SQLiteLearningUnitRepository

RULE_BASED_TYPES = frozenset(
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
SUPPORTED_TYPES = RULE_BASED_TYPES

_ONSETS = (
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
_VOWELS = (
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
_CODAS = (
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
_SIMILAR_GROUPS = (
    ("PLAIN_ASPIRATED", "ㄱ", "ㅋ"),
    ("PLAIN_TENSE", "ㄱ", "ㄲ"),
    ("PLAIN_ASPIRATED", "ㄷ", "ㅌ"),
    ("PLAIN_TENSE", "ㄷ", "ㄸ"),
    ("PLAIN_ASPIRATED", "ㅂ", "ㅍ"),
    ("PLAIN_TENSE", "ㅂ", "ㅃ"),
    ("PLAIN_ASPIRATED", "ㅈ", "ㅊ"),
    ("PLAIN_TENSE", "ㅈ", "ㅉ"),
    ("PLAIN_TENSE", "ㅅ", "ㅆ"),
)
_REPLACEMENTS = (
    ("나무", "나비", 1),
    ("바다", "바지", 1),
    ("모자", "과자", 0),
    ("기차", "기린", 1),
    ("다리", "머리", 0),
)


class RuleBasedBasicTrainingGenerator:
    def __init__(self, repository: SQLiteLearningUnitRepository) -> None:
        self._repository = repository

    @property
    def repository(self) -> SQLiteLearningUnitRepository:
        return self._repository

    def generate(
        self,
        request: TrainingCandidateRequest,
    ) -> TrainingCandidateResponse | None:
        if request.trainingType not in RULE_BASED_TYPES:
            return None
        units = self._repository.find_all_active()
        variant = int.from_bytes(
            hashlib.sha256(request.requestId.encode("utf-8")).digest()[:2],
            "big",
        )
        candidates: list[dict[str, object]] = []
        canonical: set[str] = set()
        for attempt in range(request.count * 80):
            candidate = self._candidate(request, units, variant + attempt)
            if candidate is None:
                return None
            signature = repr(candidate)
            if signature not in canonical:
                canonical.add(signature)
                candidates.append(candidate)
            if len(candidates) == request.count:
                return TrainingCandidateResponse(type=request.trainingType, data=candidates)
        return None

    def _candidate(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
    ) -> dict[str, object] | None:
        training_type = request.trainingType
        eligible = self._eligible(request, units, training_type)
        if not eligible:
            return None
        correct = eligible[variant % len(eligible)]
        if training_type in {"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"}:
            return self._trace(training_type, correct, variant)
        if training_type in {"CONSONANT_SOUND_CHOICE", "VOWEL_SOUND_CHOICE"}:
            options = [unit for unit in units if unit.unit_type == correct.unit_type]
            return self._choice(
                correct, correct.surface, options, lambda unit: unit.surface, units, variant
            )
        if training_type == "CONSONANT_VOWEL_CLASSIFICATION":
            choices = ["CONSONANT", "VOWEL"]
            answer = "CONSONANT" if correct.unit_type == "CONSONANT" else "VOWEL"
            if variant % 2:
                choices.reverse()
            audio_variants = (
                correct.spoken_text,
                f"{correct.spoken_text}, {correct.spoken_text}",
                f"{correct.spoken_text} 소리",
                f"{correct.spoken_text}를 들어요",
                f"천천히 {correct.spoken_text}",
            )
            return {
                "audioText": audio_variants[variant % len(audio_variants)],
                "choices": choices,
                "answerIndex": choices.index(answer),
            }
        if training_type in {"SYLLABLE_INITIAL_CHOICE", "WORD_INITIAL_CHOICE"}:
            options = [unit for unit in units if unit.unit_type == "CONSONANT"]
            return self._choice(
                correct, correct.onset, options, lambda unit: unit.surface, units, variant
            )
        if training_type == "SAME_INITIAL_WORD_CHOICE":
            word_options = [unit for unit in units if unit.unit_type == "WORD"]
            correct_word = next(
                (
                    unit
                    for unit in self._rotate(word_options, variant)
                    if unit.onset == correct.onset
                ),
                None,
            )
            if correct_word is None:
                return None
            choices = self._choices(
                correct_word,
                correct_word.surface,
                word_options,
                lambda unit: unit.surface,
                units,
                variant,
                predicate=lambda unit: unit.onset != correct.onset,
            )
            return {
                "targetType": "WORD",
                "targetAudioText": correct.surface,
                "choiceType": "WORD",
                "choices": [{"text": value, "imagePrompt": ""} for value in choices],
                "answerIndex": choices.index(correct_word.surface),
            }
        if training_type in {"FINAL_CONSONANT_CHOICE", "WORD_FINAL_SOUND_CHOICE"}:
            options = [
                unit
                for unit in units
                if unit.unit_type == "CONSONANT"
                and any(code.startswith("GRAPHEME.CODA.") for code in unit.feature_codes)
            ]
            return self._choice(
                correct, correct.coda, options, lambda unit: unit.surface, units, variant
            )
        if training_type == "FINAL_CONSONANT_COMPARISON":
            options = [
                unit
                for unit in units
                if unit.unit_type == "SYLLABLE"
                and unit.coda
                and unit.onset == correct.onset
                and unit.vowel == correct.vowel
            ]
            return self._choice(
                correct, correct.surface, options, lambda unit: unit.surface, units, variant
            )
        if training_type == "SIMILAR_SOUND_CHOICE":
            matching_groups = [
                entry for entry in _SIMILAR_GROUPS if correct.surface in entry[1:]
            ]
            if not matching_groups:
                return None
            group, first, second = matching_groups[variant % len(matching_groups)]
            answer = correct.surface
            contrasted = second if answer == first else first
            choices = [answer, contrasted]
            if variant % 2:
                choices.reverse()
            vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[variant % 5]
            return {
                "soundGroup": group,
                "audioText": self._compose(answer, vowel),
                "choices": choices,
                "answerIndex": choices.index(answer),
            }
        if training_type == "PHONEME_BLEND":
            parts = [correct.onset, correct.vowel, *([correct.coda] if correct.coda else [])]
            distractor = self._different(_ONSETS if variant % 2 else _VOWELS, parts, variant)
            cards = self._rotate([*parts, distractor], variant)
            return {
                "audioParts": parts,
                "cards": cards,
                "answerOrder": [cards.index(part) for part in parts],
                "result": correct.surface,
            }
        if training_type == "SYLLABLE_BLEND":
            parts = list(correct.surface)
            distractor = self._different(
                [unit.surface for unit in units if unit.unit_type == "SYLLABLE"],
                parts,
                variant,
            )
            cards = self._rotate([*parts, distractor], variant)
            return {
                "audioParts": parts,
                "cards": cards,
                "answerOrder": [cards.index(part) for part in parts],
                "result": correct.surface,
            }
        if training_type in {"BASIC_SYLLABLE_BUILD", "FINAL_SYLLABLE_BUILD", "DOUBLE_FINAL_BUILD"}:
            return self._syllable_build(training_type, correct, variant)
        if training_type == "FINAL_CONSONANT_DELETE":
            result = self._compose(correct.onset or "ㅇ", correct.vowel or "ㅏ")
            return {
                "source": correct.surface,
                "targetAudioText": result,
                "removableUnits": [correct.onset, correct.vowel, correct.coda],
                "answerIndex": 2,
                "result": result,
            }
        if training_type == "SYLLABLE_DELETE":
            index = variant % len(correct.surface)
            result = correct.surface[:index] + correct.surface[index + 1 :]
            return {
                "source": correct.surface,
                "targetAudioText": result,
                "syllables": list(correct.surface),
                "deleteIndex": index,
                "result": result,
            }
        if training_type == "SYLLABLE_REPLACE":
            source, result, index = _REPLACEMENTS[variant % len(_REPLACEMENTS)]
            answer = result[index]
            choices = self._rotate([answer, "가", "도"], variant)
            return {
                "source": source,
                "targetAudioText": result,
                "replaceIndex": index,
                "choices": choices,
                "answerIndex": choices.index(answer),
                "result": result,
            }
        if training_type == "WORD_READING":
            words = self._word_set(
                correct,
                [unit for unit in units if unit.unit_type == "WORD"],
                variant,
                3,
            )
            return {"readingOrder": "SEQUENTIAL", "words": words}
        if training_type == "NONWORD_READING":
            words = [unit.surface for unit in units if unit.unit_type == "WORD"]
            first = correct.surface[0]
            second = words[(variant + 1) % len(words)][-1]
            nonword = first + second
            while nonword in words:
                second = words[(variant + len(nonword) + 3) % len(words)][-1]
                nonword = first + second
            return {
                "words": [
                    {"text": correct.surface, "isNonword": False},
                    {"text": nonword, "isNonword": True},
                ]
            }
        if training_type == "WORD_CHAIN_READING":
            words = self._word_set(
                correct,
                [unit for unit in units if unit.unit_type == "WORD"],
                variant,
                4,
            )
            return {"words": words, "requiredOrder": "SEQUENTIAL"}
        return None

    def _eligible(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        training_type: str,
    ) -> list[LearningUnit]:
        types = {
            "VOWEL_TRACE": {"VOWEL"},
            "CONSONANT_TRACE": {"CONSONANT"},
            "SYLLABLE_TRACE": {"SYLLABLE"},
            "CONSONANT_SOUND_CHOICE": {"CONSONANT"},
            "VOWEL_SOUND_CHOICE": {"VOWEL"},
            "CONSONANT_VOWEL_CLASSIFICATION": {"CONSONANT", "VOWEL"},
            "SYLLABLE_INITIAL_CHOICE": {"SYLLABLE"},
            "WORD_INITIAL_CHOICE": {"WORD"},
            "SAME_INITIAL_WORD_CHOICE": {"WORD"},
            "FINAL_CONSONANT_CHOICE": {"SYLLABLE"},
            "WORD_FINAL_SOUND_CHOICE": {"WORD"},
            "FINAL_CONSONANT_COMPARISON": {"SYLLABLE"},
            "SIMILAR_SOUND_CHOICE": {"CONSONANT"},
            "PHONEME_BLEND": {"SYLLABLE"},
            "SYLLABLE_BLEND": {"WORD"},
            "BASIC_SYLLABLE_BUILD": {"SYLLABLE"},
            "FINAL_SYLLABLE_BUILD": {"SYLLABLE"},
            "DOUBLE_FINAL_BUILD": {"SYLLABLE"},
            "FINAL_CONSONANT_DELETE": {"SYLLABLE"},
            "SYLLABLE_DELETE": {"WORD"},
            "SYLLABLE_REPLACE": {"WORD"},
            "WORD_READING": {"WORD"},
            "NONWORD_READING": {"WORD"},
            "WORD_CHAIN_READING": {"WORD"},
        }[training_type]
        difficulty_limit = max(
            request.difficulty,
            4 if training_type == "DOUBLE_FINAL_BUILD" else 1,
        )
        targets = [feature.featureCode for feature in request.targetFeatures]
        result = [
            unit
            for unit in units
            if unit.unit_type in types
            and unit.difficulty <= difficulty_limit
            and not any(unit.matches_feature(code) for code in request.excludedFeatures)
            and (not targets or all(unit.matches_feature(code) for code in targets))
        ]
        result = [unit for unit in result if self._required_shape(training_type, unit)]
        ordered = sorted(result, key=lambda unit: (-unit.familiarity, unit.difficulty, unit.id))
        if targets and not ordered:
            return []
        return ordered

    @staticmethod
    def _required_shape(training_type: str, unit: LearningUnit) -> bool:
        if training_type in {"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"}:
            return unit.trace_asset_key is not None
        if training_type in {
            "FINAL_CONSONANT_CHOICE",
            "WORD_FINAL_SOUND_CHOICE",
            "FINAL_CONSONANT_COMPARISON",
            "FINAL_CONSONANT_DELETE",
        }:
            return unit.coda is not None
        if training_type == "BASIC_SYLLABLE_BUILD":
            return unit.coda is None
        if training_type == "FINAL_SYLLABLE_BUILD":
            return unit.coda is not None and unit.coda not in COMPLEX_CODA_PARTS
        if training_type == "DOUBLE_FINAL_BUILD":
            return unit.coda in COMPLEX_CODA_PARTS
        if training_type == "SIMILAR_SOUND_CHOICE":
            return any(unit.surface in entry[1:] for entry in _SIMILAR_GROUPS)
        if training_type in {"SYLLABLE_DELETE", "SYLLABLE_REPLACE", "SYLLABLE_BLEND"}:
            return len(unit.surface) >= 2
        return True

    @staticmethod
    def _trace(
        training_type: str,
        unit: LearningUnit,
        variant: int,
    ) -> dict[str, object]:
        sound_variants = (
            unit.spoken_text,
            f"{unit.spoken_text}, {unit.spoken_text}",
            f"{unit.spoken_text} 소리",
            f"{unit.spoken_text}를 따라 써요",
            f"천천히 {unit.spoken_text}",
        )
        sound_text = sound_variants[variant % len(sound_variants)]
        if training_type == "VOWEL_TRACE":
            kind = (
                "COMPLEX" if "GRAPHEME.VOWEL.COMPOUND" in ".".join(unit.feature_codes) else "BASIC"
            )
            return {
                "vowelType": kind,
                "target": unit.surface,
                "soundText": sound_text,
                "traceAssetKey": unit.trace_asset_key,
            }
        if training_type == "CONSONANT_TRACE":
            kind = (
                "TENSE"
                if "ONSET.TENSE" in ".".join(unit.feature_codes)
                else "ASPIRATED"
                if "ONSET.ASPIRATED" in ".".join(unit.feature_codes)
                else "BASIC"
            )
            return {
                "consonantType": kind,
                "target": unit.surface,
                "soundText": sound_text,
                "traceAssetKey": unit.trace_asset_key,
            }
        kind = "WITH_FINAL" if unit.coda else "WITHOUT_FINAL"
        if unit.coda in COMPLEX_CODA_PARTS:
            kind = "WITH_FINAL"
        return {
            "syllableType": kind,
            "target": unit.surface,
            "soundText": sound_text,
            "traceAssetKey": unit.trace_asset_key,
        }

    def _choice(
        self,
        correct_unit: LearningUnit,
        correct: str | None,
        options: list[LearningUnit],
        value: Callable[[LearningUnit], str],
        units: tuple[LearningUnit, ...],
        variant: int,
    ) -> dict[str, object] | None:
        if correct is None:
            return None
        choices = self._choices(correct_unit, correct, options, value, units, variant)
        if len(choices) != 3:
            return None
        return {
            "audioText": correct_unit.spoken_text,
            "choices": choices,
            "answerIndex": choices.index(correct),
        }

    def _choices(
        self,
        correct_unit: LearningUnit,
        correct: str,
        options: list[LearningUnit],
        value: Callable[[LearningUnit], str],
        units: tuple[LearningUnit, ...],
        variant: int,
        predicate: Callable[[LearningUnit], bool] | None = None,
    ) -> list[str]:
        by_id = {unit.id: unit for unit in units}
        allowed = {unit.id for unit in options}
        distractors: list[str] = []
        for unit_id in correct_unit.confusion_ids:
            unit = by_id.get(unit_id)
            if unit is not None and unit.id in allowed and (predicate is None or predicate(unit)):
                self._append_unique(distractors, value(unit), correct)
        for unit in self._rotate(options, variant):
            if predicate is None or predicate(unit):
                self._append_unique(distractors, value(unit), correct)
        if len(distractors) < 2:
            return []
        choices = [
            correct,
            distractors[variant % len(distractors)],
            distractors[(variant + 1) % len(distractors)],
        ]
        if choices[1] == choices[2]:
            choices[2] = distractors[(variant + 2) % len(distractors)]
        return self._rotate(choices, variant)

    def _syllable_build(
        self, training_type: str, unit: LearningUnit, variant: int
    ) -> dict[str, object]:
        initial = unit.onset or "ㅇ"
        medial = unit.vowel or "ㅏ"
        initial_choices = self._option_values(initial, _ONSETS, variant)
        medial_choices = self._option_values(medial, _VOWELS, variant + 1)
        result: dict[str, object] = {
            "targetAudioText": unit.surface,
            "initialChoices": initial_choices,
            "medialChoices": medial_choices,
            "initialAnswerIndex": initial_choices.index(initial),
            "medialAnswerIndex": medial_choices.index(medial),
            "result": unit.surface,
        }
        if training_type != "BASIC_SYLLABLE_BUILD":
            final = unit.coda or "ㄱ"
            final_choices = self._option_values(final, _CODAS[1:], variant + 2)
            result["finalChoices"] = final_choices
            result["finalAnswerIndex"] = final_choices.index(final)
        return result

    @staticmethod
    def _option_values(correct: str, pool: Iterable[str], variant: int) -> list[str]:
        others = [value for value in pool if value != correct]
        first = others[variant % len(others)]
        second = others[(variant + 3) % len(others)]
        values = [correct, first, second]
        return RuleBasedBasicTrainingGenerator._rotate(values, variant)

    @staticmethod
    def _different(pool: Iterable[str], excluded: Iterable[str], variant: int) -> str:
        values = [value for value in pool if value not in set(excluded)]
        return values[variant % len(values)]

    @classmethod
    def _word_set(
        cls,
        correct: LearningUnit,
        eligible: list[LearningUnit],
        variant: int,
        count: int,
    ) -> list[str]:
        words = [correct.surface]
        for unit in cls._rotate(eligible, variant):
            if unit.surface not in words:
                words.append(unit.surface)
            if len(words) == count:
                break
        return words

    @staticmethod
    def _compose(onset: str, vowel: str, coda: str = "") -> str:
        return chr(
            0xAC00 + _ONSETS.index(onset) * 588 + _VOWELS.index(vowel) * 28 + _CODAS.index(coda)
        )

    @staticmethod
    def _append_unique(values: list[str], candidate: str, correct: str) -> None:
        if candidate != correct and candidate not in values:
            values.append(candidate)

    @staticmethod
    def _rotate(values: Iterable, amount: int) -> list:
        result = list(values)
        if not result:
            return result
        offset = amount % len(result)
        return result[-offset:] + result[:-offset] if offset else result


__all__ = ["RULE_BASED_TYPES", "RuleBasedBasicTrainingGenerator", "SUPPORTED_TYPES"]
