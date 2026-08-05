from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable

from iread_ai.generation_models import TrainingCandidateRequest, TrainingCandidateResponse
from iread_ai.personalization.hangul import COMPLEX_CODA_PARTS, decompose_text
from iread_ai.training_feature_compatibility import compatible_features

from .models import LearningUnit
from .repository import SQLiteLearningUnitRepository
from .seed import UnitSeed, unit_features, unit_parts

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
        "FILL_IN_THE_BLANK",
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
_SIMPLE_CODAS = ("ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ")
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
_TENSE_ONSETS = ("ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ")
_COMPOUND_VOWELS = ("ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ")
_COMPLEX_CODAS = ("ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅄ")
_MECHANICAL_WORDS = (
    "나무",
    "바다",
    "토끼",
    "모자",
    "사과",
    "나비",
    "강아지",
    "무지개",
    "해바라기",
    "아기고양이",
    "의자",
    "과자",
    "돼지",
)
_REPLACEMENT_SYLLABLES = ("가", "나", "도", "라", "미", "보", "수", "지")

_FILL_BLANK_ITEMS = (
    ("아이는 {{blank}}를 먹어요.", "포도", ("우표", "모자")),
    ("편지에 {{blank}}를 붙여요.", "우표", ("포도", "모자")),
    ("화분에 {{blank}}를 심어요.", "씨", ("해", "비")),
    ("아빠가 {{blank}}를 보내요.", "소포", ("모자", "나무")),
    ("파도가 {{blank}}를 적셔요.", "모래", ("바다", "나비")),
    ("아이가 {{blank}}를 써요.", "연필", ("수박", "구름")),
    ("새가 {{blank}}에서 노래해요.", "나무", ("바다", "버스")),
    ("비가 오면 {{blank}}을 써요.", "우산", ("연필", "가방")),
    ("여름에 {{blank}}을 시원하게 먹어요.", "수박", ("연필", "가방")),
    ("깡충 뛰는 {{blank}}와 놀아요.", "토끼", ("여우", "나비")),
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
        units = (
            *self._lexicon_word_units(request),
            *self._repository.find_all_active(),
        )
        variant = int.from_bytes(
            hashlib.sha256(request.requestId.encode("utf-8")).digest()[:2],
            "big",
        )
        candidates: list[dict[str, object]] = []
        canonical: set[str] = set()
        for slot in range(request.count):
            slot_request = request.model_copy(
                update={"targetFeatures": self._targets_for_slot(request, slot)}
            )
            for attempt in range(80):
                candidate = self._candidate(
                    slot_request,
                    units,
                    variant + (slot * 80) + attempt,
                    slot,
                )
                if candidate is None:
                    return None
                signature = repr(candidate)
                if signature in canonical:
                    continue
                canonical.add(signature)
                candidates.append(candidate)
                break
            else:
                return None
        return TrainingCandidateResponse(type=request.trainingType, data=candidates)

    @staticmethod
    def _lexicon_word_units(
        request: TrainingCandidateRequest,
    ) -> tuple[LearningUnit, ...]:
        features_by_surface: dict[str, set[str]] = {}
        ordered_surfaces: list[str] = []
        for feature_code, words in request.recommendedWordsByFeature.items():
            for word in words:
                surface = word.strip()
                if re.fullmatch(r"[가-힣]+", surface) is None:
                    continue
                if surface not in features_by_surface:
                    features_by_surface[surface] = set()
                    ordered_surfaces.append(surface)
                if feature_code != "__DEFAULT__":
                    features_by_surface[surface].add(feature_code)

        units: list[LearningUnit] = []
        for surface in ordered_surfaces:
            seed = UnitSeed(
                unit_type="WORD",
                surface=surface,
                spoken_text=surface,
                pronunciation=surface,
                difficulty=request.difficulty,
                familiarity=10,
            )
            onset, vowel, coda = unit_parts(seed)
            features = set(unit_features(seed)) | features_by_surface[surface]
            units.append(
                LearningUnit(
                    id=-(len(units) + 1),
                    unit_type="WORD",
                    surface=surface,
                    spoken_text=surface,
                    pronunciation=surface,
                    onset=onset,
                    vowel=vowel,
                    coda=coda,
                    difficulty=request.difficulty,
                    familiarity=10,
                    trace_asset_key=None,
                    feature_codes=frozenset(features),
                    confusion_ids=(),
                )
            )
        return tuple(units)

    @staticmethod
    def _targets_for_slot(
        request: TrainingCandidateRequest,
        slot: int,
    ) -> list:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        if request.trainingType in {"WORD_READING", "WORD_CHAIN_READING"}:
            return targets
        if len(targets) <= 1:
            return targets
        primary_count = (request.count + 1) // 2
        return [targets[0] if slot < primary_count else targets[1]]

    def _candidate(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
        slot: int,
    ) -> dict[str, object] | None:
        training_type = request.trainingType
        if training_type == "FILL_IN_THE_BLANK":
            return self._fill_blank(request, variant, slot)
        if training_type in {"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"}:
            return self._trace_candidate(request, units, training_type, variant, slot)
        if training_type == "CONSONANT_VOWEL_CLASSIFICATION":
            return self._classification(request, variant, slot)
        if training_type == "CONSONANT_SOUND_CHOICE":
            return self._consonant_sound_choice(request, variant)
        if training_type == "SYLLABLE_INITIAL_CHOICE":
            return self._syllable_initial_choice(request, variant, slot)
        if training_type == "FINAL_CONSONANT_CHOICE":
            return self._final_consonant_choice(request, variant, slot)
        if training_type == "WORD_FINAL_SOUND_CHOICE":
            return self._word_final_sound_choice(request, variant, slot)
        if training_type == "FINAL_CONSONANT_COMPARISON":
            return self._final_consonant_comparison(request, variant, slot)
        if training_type == "SIMILAR_SOUND_CHOICE":
            return self._similar_sound_choice(request, variant, slot)
        if training_type == "SYLLABLE_BLEND":
            return self._syllable_blend(request, units, variant, slot)
        if training_type == "BASIC_SYLLABLE_BUILD":
            return self._basic_syllable_build(request, variant)
        if training_type == "FINAL_SYLLABLE_BUILD":
            return self._final_syllable_build(request, units, variant)
        if training_type == "DOUBLE_FINAL_BUILD":
            return self._double_final_build(request, variant)
        if training_type == "FINAL_CONSONANT_DELETE":
            return self._final_consonant_delete(request, variant, slot)
        if training_type == "PHONEME_BLEND":
            return self._phoneme_blend(request, variant, slot)
        if training_type == "SYLLABLE_DELETE":
            return self._syllable_delete(request, units, variant, slot)
        if training_type == "SYLLABLE_REPLACE":
            return self._syllable_replace(request, units, variant, slot)
        palette_candidate = self._word_palette_candidate(
            request,
            training_type,
            variant,
            units,
        )
        if palette_candidate is not None:
            return palette_candidate
        eligible = self._eligible(request, units, training_type)
        if not eligible:
            return None
        correct = eligible[variant % len(eligible)]
        if training_type in {"CONSONANT_SOUND_CHOICE", "VOWEL_SOUND_CHOICE"}:
            options = [unit for unit in units if unit.unit_type == correct.unit_type]
            candidate = self._choice(
                correct, correct.surface, options, lambda unit: unit.surface, units, variant
            )
            if candidate is not None:
                candidate["audioText"] = correct.surface
            return candidate
        if training_type in {"SYLLABLE_INITIAL_CHOICE", "WORD_INITIAL_CHOICE"}:
            options = [unit for unit in units if unit.unit_type == "CONSONANT"]
            return self._choice(
                correct, correct.onset, options, lambda unit: unit.surface, units, variant
            )
        if training_type == "SAME_INITIAL_WORD_CHOICE":
            word_options = [unit for unit in units if unit.unit_type == "WORD"]
            answer_options = [unit for unit in word_options if unit.id < 0] or word_options
            correct_word = next(
                (
                    unit
                    for unit in self._rotate(answer_options, variant)
                    if unit.onset == correct.onset and unit.surface != correct.surface
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
        if training_type == "WORD_FINAL_SOUND_CHOICE":
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
        if training_type == "WORD_READING":
            words = self._word_set(
                correct,
                [unit for unit in units if unit.unit_type == "WORD"],
                variant,
                4,
            )
            return {"readingOrder": "SEQUENTIAL", "words": words}
        if training_type == "NONWORD_READING":
            words = list(dict.fromkeys(unit.surface for unit in eligible))
            real_words = self._rotate(words, variant)
            selected_real = [correct.surface]
            for word in real_words:
                if word != correct.surface:
                    selected_real.append(word)
                    break
            nonwords = self._nonwords(words, correct.surface, variant, 2)
            if len(selected_real) != 2 or len(nonwords) != 2:
                return None
            return {
                "words": [
                    {"text": selected_real[0], "isNonword": False},
                    {"text": nonwords[0], "isNonword": True},
                    {"text": selected_real[1], "isNonword": False},
                    {"text": nonwords[1], "isNonword": True},
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

    @staticmethod
    def _classification(
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        target = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _ONSETS + _VOWELS
            ),
            "ㄱ",
        )
        target_pool = _ONSETS if target in _ONSETS else _VOWELS
        contrast_pool = _VOWELS if target in _ONSETS else _ONSETS
        if slot in {0, 3}:
            glyph = target
        elif slot % 2:
            glyph = contrast_pool[(variant + slot) % len(contrast_pool)]
        else:
            alternatives = [value for value in target_pool if value != target]
            glyph = alternatives[(variant + slot) % len(alternatives)]
        answer = "CONSONANT" if glyph in _ONSETS else "VOWEL"
        choices = ["CONSONANT", "VOWEL"]
        if slot % 2:
            choices.reverse()
        return {
            "audioText": glyph,
            "choices": choices,
            "answerIndex": choices.index(answer),
        }

    @staticmethod
    def _consonant_sound_choice(
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        answer = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _ONSETS
            ),
            _ONSETS[variant % len(_ONSETS)],
        )
        distractors = [value for value in _ONSETS if value != answer]
        choices = [
            answer,
            distractors[variant % len(distractors)],
            distractors[(variant + 5) % len(distractors)],
        ]
        choices = RuleBasedBasicTrainingGenerator._rotate(choices, variant)
        return {
            "audioText": answer,
            "choices": choices,
            "answerIndex": choices.index(answer),
        }

    def _trace_candidate(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        training_type: str,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        compatible = compatible_features(training_type, request.targetFeatures)
        trace_request = request.model_copy(update={"targetFeatures": compatible})
        eligible = self._eligible(trace_request, units, training_type)
        if not eligible:
            eligible = self._eligible(
                trace_request.model_copy(update={"difficulty": 5}),
                units,
                training_type,
            )
        if not eligible:
            unit_type = {
                "VOWEL_TRACE": "VOWEL",
                "CONSONANT_TRACE": "CONSONANT",
                "SYLLABLE_TRACE": "SYLLABLE",
            }[training_type]
            eligible = [
                unit
                for unit in units
                if unit.unit_type == unit_type and unit.trace_asset_key is not None
            ]
        unit = eligible[slot % len(eligible)]
        return self._trace(training_type, unit, slot)

    @classmethod
    def _syllable_initial_choice(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        onset = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _ONSETS
            ),
            _ONSETS[(variant + slot) % len(_ONSETS)],
        )
        vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[(variant + slot) % 5]
        audio_text = cls._compose(onset, vowel)
        choices = cls._option_values(onset, _ONSETS, variant + slot)
        return {
            "audioText": audio_text,
            "choices": choices,
            "answerIndex": choices.index(onset),
        }

    @classmethod
    def _final_consonant_choice(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        coda = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _SIMPLE_CODAS
            ),
            _SIMPLE_CODAS[slot % len(_SIMPLE_CODAS)],
        )
        onsets = ("ㄱ", "ㄴ", "ㄷ", "ㅁ", "ㅂ")
        vowels = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")
        syllable = cls._compose(onsets[slot % len(onsets)], vowels[slot % len(vowels)], coda)
        choice_pool = (coda, *[value for value in _SIMPLE_CODAS if value != coda])
        choices = cls._option_values(coda, choice_pool, variant + slot)
        return {
            "audioText": syllable,
            "choices": choices,
            "answerIndex": choices.index(coda),
        }

    @classmethod
    def _word_final_sound_choice(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        coda = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _SIMPLE_CODAS
            ),
            _SIMPLE_CODAS[slot % len(_SIMPLE_CODAS)],
        )
        palette = [
            word
            for words in request.recommendedWordsByFeature.values()
            for word in words
        ]
        palette.extend(request.recommendedWords)
        matching_words = [
            word
            for word in dict.fromkeys(palette)
            if re.fullmatch(r"[가-힣]+", word)
            and decompose_text(word)[-1].coda == coda
        ]
        if matching_words:
            word = matching_words[(variant + slot) % len(matching_words)]
        else:
            onset = ("ㄱ", "ㄴ", "ㅁ", "ㅂ", "ㅅ")[(variant + slot) % 5]
            vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[(variant + slot) % 5]
            word = cls._compose(onset, vowel, coda)
        choices = cls._option_values(coda, _SIMPLE_CODAS, variant + slot)
        return {
            "audioText": word,
            "choices": choices,
            "answerIndex": choices.index(coda),
        }

    @classmethod
    def _final_consonant_comparison(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        coda = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _SIMPLE_CODAS
            ),
            _SIMPLE_CODAS[slot % len(_SIMPLE_CODAS)],
        )
        onset = ("ㄱ", "ㄴ", "ㅁ", "ㅂ", "ㅅ")[(variant + slot) % 5]
        vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[(variant + slot) % 5]
        codas = cls._option_values(coda, _SIMPLE_CODAS, variant + slot)
        choices = [cls._compose(onset, vowel, value) for value in codas]
        answer = cls._compose(onset, vowel, coda)
        return {
            "audioText": answer,
            "choices": choices,
            "answerIndex": choices.index(answer),
        }

    @classmethod
    def _similar_sound_choice(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        requested = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if any(
                    feature.featureCode.rsplit(".", 1)[-1] in pair[1:]
                    for pair in _SIMILAR_GROUPS
                )
            ),
            None,
        )
        matching = [pair for pair in _SIMILAR_GROUPS if requested in pair[1:]]
        groups = matching or list(_SIMILAR_GROUPS)
        group, first, second = groups[(variant + slot) % len(groups)]
        answer = requested if requested in {first, second} else (first, second)[slot % 2]
        choices = [first, second]
        if variant % 2:
            choices.reverse()
        vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[(variant + slot) % 5]
        return {
            "soundGroup": group,
            "audioText": cls._compose(answer, vowel),
            "choices": choices,
            "answerIndex": choices.index(answer),
        }

    def _syllable_blend(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        palette = [
            word
            for words in request.recommendedWordsByFeature.values()
            for word in words
        ]
        palette.extend(request.recommendedWords)

        def valid(word: str) -> bool:
            return re.fullmatch(r"[가-힣]{2,4}", word) is not None and len(set(word)) == len(word)

        preferred_words = list(dict.fromkeys(word for word in palette if valid(word)))
        fallback_words = list(
            dict.fromkeys(
                unit.surface
                for unit in units
                if unit.unit_type == "WORD" and valid(unit.surface)
            )
        )
        words = preferred_words or fallback_words
        word = words[(variant + slot) % len(words)]
        parts = list(word)
        distractors = (
            "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"
        )
        distractor = next(
            value for value in self._rotate(distractors, variant) if value not in parts
        )
        cards = self._rotate([*parts, distractor], variant + slot)
        return {
            "audioParts": parts,
            "cards": cards,
            "answerOrder": [cards.index(part) for part in parts],
            "result": word,
        }

    def _basic_syllable_build(
        self,
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        onset = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _ONSETS
            ),
            _ONSETS[variant % len(_ONSETS)],
        )
        vowel = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _VOWELS
            ),
            ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[variant % 5],
        )
        return self._syllable_build_parts(
            "BASIC_SYLLABLE_BUILD", onset, vowel, "", variant
        )

    def _double_final_build(
        self,
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        coda = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _COMPLEX_CODAS
            ),
            _COMPLEX_CODAS[variant % len(_COMPLEX_CODAS)],
        )
        onset = ("ㄱ", "ㄴ", "ㅁ", "ㅂ", "ㅅ")[variant % 5]
        vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[variant % 5]
        return self._syllable_build_parts(
            "DOUBLE_FINAL_BUILD", onset, vowel, coda, variant
        )

    @classmethod
    def _final_consonant_delete(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = compatible_features(request.trainingType, request.targetFeatures)
        coda = next(
            (
                feature.featureCode.rsplit(".", 1)[-1]
                for feature in targets
                if feature.featureCode.rsplit(".", 1)[-1] in _SIMPLE_CODAS
            ),
            _SIMPLE_CODAS[slot % len(_SIMPLE_CODAS)],
        )
        onset = ("ㄱ", "ㄴ", "ㅁ", "ㅂ", "ㅅ")[(variant + slot) % 5]
        vowel = ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[(variant + slot) % 5]
        source = cls._compose(onset, vowel, coda)
        result = cls._compose(onset, vowel)
        return {
            "source": source,
            "targetAudioText": result,
            "removableUnits": [onset, vowel, coda],
            "answerIndex": 2,
            "result": result,
        }

    @staticmethod
    def _fill_blank(
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        del request
        sentence, answer, distractors = _FILL_BLANK_ITEMS[slot % len(_FILL_BLANK_ITEMS)]
        choices = RuleBasedBasicTrainingGenerator._rotate(
            [answer, *distractors], variant + slot
        )
        return {
            "sentence": sentence,
            "inputType": "CHOICE",
            "choices": choices,
            "answerIndex": choices.index(answer),
            "acceptedAnswers": [answer],
            "completedSentence": sentence.replace("{{blank}}", answer),
        }

    @classmethod
    def _phoneme_blend(
        cls,
        request: TrainingCandidateRequest,
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        targets = [
            feature.featureCode
            for feature in compatible_features(request.trainingType, request.targetFeatures)
        ]
        onset = _ONSETS[(variant + slot) % len(_ONSETS)]
        vowel = _VOWELS[(variant + slot * 3) % len(_VOWELS)]
        coda: str | None = None
        for target in targets:
            if target.startswith("GRAPHEME.ONSET."):
                onset = target.rsplit(".", 1)[-1]
            elif target.startswith("GRAPHEME.VOWEL."):
                vowel = target.rsplit(".", 1)[-1]
            elif target.startswith("GRAPHEME.CODA."):
                coda = target.rsplit(".", 1)[-1]
            elif target == "SYLLABLE.TENSE_ONSET":
                onset = _TENSE_ONSETS[(variant + slot) % len(_TENSE_ONSETS)]
            elif target == "SYLLABLE.COMPLEX_VOWEL":
                vowel = _COMPOUND_VOWELS[(variant + slot) % len(_COMPOUND_VOWELS)]
            elif target == "SYLLABLE.COMPLEX_CODA":
                coda = _COMPLEX_CODAS[(variant + slot) % len(_COMPLEX_CODAS)]
            elif target == "SYLLABLE.CVC":
                coda = _SIMPLE_CODAS[(variant + slot) % len(_SIMPLE_CODAS)]
            elif target == "SYLLABLE.CV":
                coda = None
        if coda == onset:
            onset = next(value for value in _ONSETS if value != coda)
        result = cls._compose(onset, vowel, coda) if coda else cls._compose(onset, vowel)
        parts = [onset, vowel, *([coda] if coda else [])]
        distractor_pool = _VOWELS if slot % 2 == 0 else _ONSETS
        distractor = cls._different(distractor_pool, parts, variant + slot)
        cards = cls._rotate([*parts, distractor], variant + slot)
        unused_indices = list(range(len(cards)))
        answer_order: list[int] = []
        for part in parts:
            index = next(index for index in unused_indices if cards[index] == part)
            answer_order.append(index)
            unused_indices.remove(index)
        return {
            "audioParts": parts,
            "cards": cards,
            "answerOrder": answer_order,
            "result": result,
        }

    @classmethod
    def _syllable_delete(
        cls,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        target_length = cls._requested_word_length(request)
        preferred_length = max(target_length or 2, 2)
        words = cls._mechanical_word_pool(request, units, preferred_length)
        source = words[(variant + slot) % len(words)]
        delete_index = (variant + slot) % len(source)
        result = source[:delete_index] + source[delete_index + 1 :]
        return {
            "source": source,
            "targetAudioText": result,
            "syllables": list(source),
            "deleteIndex": delete_index,
            "result": result,
        }

    @classmethod
    def _syllable_replace(
        cls,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
        slot: int,
    ) -> dict[str, object]:
        target_length = cls._requested_word_length(request)
        words = cls._mechanical_word_pool(request, units, max(target_length or 2, 2))
        source = words[(variant + slot) % len(words)]
        replace_index = (variant + slot) % len(source)
        replacements = [
            value
            for value in cls._rotate(_REPLACEMENT_SYLLABLES, variant + slot)
            if value != source[replace_index]
        ]
        answer = replacements[0]
        result = source[:replace_index] + answer + source[replace_index + 1 :]
        choices = cls._rotate([answer, replacements[1], replacements[2]], variant + slot)
        return {
            "source": source,
            "targetAudioText": result,
            "replaceIndex": replace_index,
            "choices": choices,
            "answerIndex": choices.index(answer),
            "result": result,
        }

    @staticmethod
    def _requested_word_length(request: TrainingCandidateRequest) -> int | None:
        for feature in request.targetFeatures:
            match = re.fullmatch(r"WORD\.SYLLABLE_COUNT\.([1-5])", feature.featureCode)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _mechanical_word_pool(
        cls,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        preferred_length: int,
    ) -> list[str]:
        target_codes = [feature.featureCode for feature in request.targetFeatures]
        focused_recommended = [
            word.strip()
            for code in target_codes
            for word in request.recommendedWordsByFeature.get(code, [])
            if re.fullmatch(r"[가-힣]+", word.strip())
        ]
        recommended = [
            word.strip()
            for words in request.recommendedWordsByFeature.values()
            for word in words
            if re.fullmatch(r"[가-힣]+", word.strip())
        ]
        bank_words = [
            unit.surface
            for unit in units
            if unit.unit_type == "WORD"
            and unit.difficulty <= request.difficulty
            and re.fullmatch(r"[가-힣]+", unit.surface)
        ]
        focused_preferred = [
            word
            for word in dict.fromkeys(focused_recommended)
            if len(word) == preferred_length
        ]
        if focused_preferred:
            return focused_preferred
        target_matched = [
            unit.surface
            for unit in units
            if unit.unit_type == "WORD"
            and len(unit.surface) == preferred_length
            and any(unit.matches_feature(code) for code in target_codes)
        ]
        if target_matched:
            return list(dict.fromkeys(target_matched))
        if any(code == "SYLLABLE.COMPLEX_VOWEL" for code in target_codes):
            compound_words = [
                word
                for word in [*recommended, *bank_words, *_MECHANICAL_WORDS]
                if len(word) == preferred_length
                and any(
                    syllable.nucleus in _COMPOUND_VOWELS
                    for syllable in decompose_text(word)
                )
            ]
            if compound_words:
                return list(dict.fromkeys(compound_words))
        recommended_preferred = [
            word for word in dict.fromkeys(recommended) if len(word) == preferred_length
        ]
        if recommended_preferred:
            return recommended_preferred
        all_words = list(dict.fromkeys([*recommended, *bank_words, *_MECHANICAL_WORDS]))
        preferred = [word for word in all_words if len(word) == preferred_length]
        usable = preferred or [word for word in all_words if len(word) >= 2]
        return usable or list(_MECHANICAL_WORDS)

    @staticmethod
    def _nonwords(
        known_words: list[str],
        target_word: str,
        variant: int,
        count: int,
    ) -> list[str]:
        known = set(known_words)
        beginnings = [target_word[0], *[word[0] for word in known_words if word]]
        endings = ("누", "보", "루", "머", "디", "푸", "쏘", "깨")
        candidates: list[str] = []
        for beginning in RuleBasedBasicTrainingGenerator._rotate(beginnings, variant):
            for ending in RuleBasedBasicTrainingGenerator._rotate(endings, variant):
                candidate = beginning + ending
                if candidate not in known and candidate not in candidates:
                    candidates.append(candidate)
                if len(candidates) == count:
                    return candidates
        return candidates

    def _word_palette_candidate(
        self,
        request: TrainingCandidateRequest,
        training_type: str,
        variant: int,
        units: tuple[LearningUnit, ...],
    ) -> dict[str, object] | None:
        if training_type not in {"WORD_READING", "WORD_CHAIN_READING"}:
            return None
        feature_codes = [feature.featureCode for feature in request.targetFeatures]
        if not feature_codes:
            feature_codes = ["__DEFAULT__"]
        palettes = {
            feature_code: list(
                dict.fromkeys(
                    word.strip()
                    for word in request.recommendedWordsByFeature.get(feature_code, [])
                    if re.fullmatch(r"[가-힣]+", word.strip())
                )
            )
            for feature_code in feature_codes
        }
        syllable_count = next(
            (
                int(feature_code.rsplit(".", 1)[-1])
                for feature_code in feature_codes
                if feature_code.startswith("WORD.SYLLABLE_COUNT.")
            ),
            None,
        )

        def allowed(word: str) -> bool:
            return syllable_count is None or len(word) == syllable_count

        curated_units = [
            unit
            for unit in units
            if unit.id > 0
            and unit.unit_type == "WORD"
            and unit.familiarity >= 4
            and unit.difficulty <= request.difficulty
            and allowed(unit.surface)
            and not any(
                unit.matches_feature(code) for code in request.excludedFeatures
            )
        ]
        curated_surfaces = {unit.surface for unit in curated_units}

        words = list(
            dict.fromkeys(
                [
                    *(
                        word
                        for feature_words in palettes.values()
                        for word in feature_words
                        if allowed(word)
                    ),
                    *(
                        word.strip()
                        for word in request.recommendedWords
                        if re.fullmatch(r"[가-힣]+", word.strip()) and allowed(word.strip())
                    ),
                ]
            )
        )
        if len(words) < 4:
            return None
        selected: list[str] = []
        practiced_features = [
            feature_code
            for feature_code in feature_codes
            if feature_code != "__DEFAULT__"
            and not feature_code.startswith("WORD.SYLLABLE_COUNT.")
        ]
        per_feature = 2 if len(practiced_features) == 1 and len(feature_codes) > 1 else 4
        if len(practiced_features) > 1:
            per_feature = 2
        for feature_code in practiced_features:
            feature_words = [word for word in palettes[feature_code] if allowed(word)]
            if not feature_words:
                return None
            preferred_words = [word for word in feature_words if word in curated_surfaces]
            fallback_words = [word for word in feature_words if word not in curated_surfaces]
            ordered_feature_words = [
                *self._rotate(preferred_words, variant),
                *self._rotate(fallback_words, variant),
            ]
            for word in ordered_feature_words:
                if word not in selected:
                    selected.append(word)
                if sum(candidate in feature_words for candidate in selected) >= per_feature:
                    break
        contrast_words = (
            [
                unit.surface
                for unit in curated_units
                if not any(unit.matches_feature(code) for code in practiced_features)
            ]
            if practiced_features
            else []
        )
        for word in self._rotate(contrast_words, variant):
            if word not in selected:
                selected.append(word)
            if len(selected) == 4:
                break
        for word in self._rotate(words, variant):
            if word not in selected:
                selected.append(word)
            if len(selected) == 4:
                break
        selected = selected[:4]
        if len(selected) < 4:
            return None
        if training_type == "WORD_READING":
            return {"readingOrder": "SEQUENTIAL", "words": selected}
        return {"words": selected, "requiredOrder": "SEQUENTIAL"}

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
            and self._target_position_matches(training_type, unit, targets)
        ]
        result = [unit for unit in result if self._required_shape(training_type, unit)]
        feature_code = targets[0] if targets else "__DEFAULT__"
        if request.recommendedWordsByFeature.get(feature_code):
            lexicon_result = [unit for unit in result if unit.id < 0]
            if lexicon_result:
                result = lexicon_result
        ordered = sorted(result, key=lambda unit: (-unit.familiarity, unit.difficulty, unit.id))
        if targets and not ordered:
            return []
        return ordered

    @staticmethod
    def _target_position_matches(
        training_type: str,
        unit: LearningUnit,
        targets: list[str],
    ) -> bool:
        for target in targets:
            expected = target.rsplit(".", 1)[-1]
            if training_type in {"WORD_INITIAL_CHOICE", "SAME_INITIAL_WORD_CHOICE"}:
                if target.startswith("GRAPHEME.ONSET.") and unit.onset != expected:
                    return False
            if training_type == "WORD_FINAL_SOUND_CHOICE":
                if target.startswith("GRAPHEME.CODA.") and unit.coda != expected:
                    return False
        return True

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
        sound_text = unit.surface
        trace_asset_key = (
            unit.trace_asset_key
            if variant == 0
            else f"{unit.trace_asset_key}_v{(variant % 5) + 1}"
        )
        if training_type == "VOWEL_TRACE":
            kind = (
                "COMPLEX" if "GRAPHEME.VOWEL.COMPOUND" in ".".join(unit.feature_codes) else "BASIC"
            )
            return {
                "vowelType": kind,
                "target": unit.surface,
                "soundText": sound_text,
                "traceAssetKey": trace_asset_key,
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
                "traceAssetKey": trace_asset_key,
            }
        kind = "WITH_FINAL" if unit.coda else "WITHOUT_FINAL"
        if unit.coda in COMPLEX_CODA_PARTS:
            kind = "WITH_FINAL"
        return {
            "syllableType": kind,
            "target": unit.surface,
            "soundText": sound_text,
            "traceAssetKey": trace_asset_key,
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
        return self._syllable_build_parts(
            training_type,
            unit.onset or "ㅇ",
            unit.vowel or "ㅏ",
            unit.coda or "",
            variant,
        )

    def _final_syllable_build(
        self,
        request: TrainingCandidateRequest,
        units: tuple[LearningUnit, ...],
        variant: int,
    ) -> dict[str, object]:
        eligible = self._eligible(request, units, "FINAL_SYLLABLE_BUILD")
        if eligible:
            return self._syllable_build(
                "FINAL_SYLLABLE_BUILD",
                eligible[variant % len(eligible)],
                variant,
            )

        target_codes = [feature.featureCode for feature in request.targetFeatures]
        onset = next(
            (
                code.rsplit(".", 1)[-1]
                for code in target_codes
                if code.startswith("GRAPHEME.ONSET.")
                and code.rsplit(".", 1)[-1] in _ONSETS
            ),
            _ONSETS[variant % len(_ONSETS)],
        )
        vowel = next(
            (
                code.rsplit(".", 1)[-1]
                for code in target_codes
                if code.startswith("GRAPHEME.VOWEL.")
                and code.rsplit(".", 1)[-1] in _VOWELS
            ),
            ("ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ")[variant % 5],
        )
        simple_codas = tuple(
            coda for coda in _CODAS[1:] if coda not in _COMPLEX_CODAS
        )
        coda = next(
            (
                code.rsplit(".", 1)[-1]
                for code in target_codes
                if code.startswith("GRAPHEME.CODA.SIMPLE.")
                and code.rsplit(".", 1)[-1] in simple_codas
            ),
            simple_codas[(variant // 5) % len(simple_codas)],
        )
        return self._syllable_build_parts(
            "FINAL_SYLLABLE_BUILD",
            onset,
            vowel,
            coda,
            variant,
        )

    def _syllable_build_parts(
        self,
        training_type: str,
        initial: str,
        medial: str,
        final: str,
        variant: int,
    ) -> dict[str, object]:
        initial_choices = self._option_values(initial, _ONSETS, variant)
        medial_choices = self._option_values(medial, _VOWELS, variant + 1)
        surface = self._compose(initial, medial, final)
        result: dict[str, object] = {
            "targetAudioText": surface,
            "initialChoices": initial_choices,
            "medialChoices": medial_choices,
            "initialAnswerIndex": initial_choices.index(initial),
            "medialAnswerIndex": medial_choices.index(medial),
            "result": surface,
        }
        if training_type != "BASIC_SYLLABLE_BUILD":
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
