from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable

from iread_ai.generation_models import TrainingCandidateRequest, TrainingCandidateResponse
from iread_ai.personalization.hangul import COMPLEX_CODA_PARTS, decompose_text
from iread_ai.training_feature_compatibility import compatible_features
from iread_ai.training_final_sounds import (
    REPRESENTATIVE_FINAL_SOUNDS,
    representative_final_sound,
)

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

_FINAL_CHOICE_WRITTEN_CODAS = REPRESENTATIVE_FINAL_SOUNDS

_FINAL_CHOICE_BASES = (
    ("ㄱ", "ㅏ"),
    ("ㄴ", "ㅓ"),
    ("ㄷ", "ㅗ"),
    ("ㄹ", "ㅜ"),
    ("ㅁ", "ㅣ"),
    ("ㅂ", "ㅏ"),
    ("ㅅ", "ㅓ"),
    ("ㅈ", "ㅗ"),
    ("ㅎ", "ㅏ"),
)

_FINAL_CHOICE_SYLLABLES: dict[str, tuple[str, ...]] = {
    "ㄱ": ("각", "목", "북", "녹", "국"),
    "ㄲ": ("밖", "엮", "볶", "닦", "묶"),
    "ㄴ": ("간", "눈", "문", "산", "손"),
    "ㄷ": ("곧", "닫", "믿", "받", "얻"),
    "ㄹ": ("달", "길", "물", "별", "돌"),
    "ㅁ": ("감", "몸", "밤", "봄", "꿈"),
    "ㅂ": ("갑", "밥", "입", "컵", "법"),
    "ㅅ": ("갓", "옷", "빗", "맛", "못"),
    "ㅆ": ("있", "났", "했", "갔", "컸"),
    "ㅇ": ("강", "공", "방", "빵", "콩"),
    "ㅈ": ("낮", "빚", "젖", "잦", "맺"),
    "ㅊ": ("꽃", "빛", "숯", "낯", "윷"),
    "ㅋ": ("녘", "엌", "옼", "앜", "읔"),
    "ㅌ": ("밭", "끝", "겉", "솥", "밑"),
    "ㅍ": ("앞", "숲", "옆", "잎", "깊"),
    "ㅎ": ("좋", "놓", "닿", "낳", "넣"),
}
_REPLACEMENTS = (
    ("나무", "나비", 1),
    ("바다", "바지", 1),
    ("모자", "과자", 0),
    ("기차", "기린", 1),
    ("다리", "머리", 0),
)

_FINAL_DELETE_SOURCES: dict[str, tuple[str, ...]] = {
    "ㄱ": ("각", "국", "목", "북", "약"),
    "ㄲ": ("밖", "볶", "섞", "깎", "엮"),
    "ㄳ": ("넋", "몫", "삯"),
    "ㄴ": ("간", "눈", "문", "산", "손"),
    "ㄵ": ("앉", "얹"),
    "ㄶ": ("많", "않", "끊"),
    "ㄷ": ("곧", "낟", "맏", "믿", "닫"),
    "ㄹ": ("달", "길", "물", "별", "돌"),
    "ㄺ": ("닭", "흙", "읽", "맑", "밝"),
    "ㄻ": ("삶", "젊", "닮", "굶", "옮"),
    "ㄼ": ("넓", "밟", "짧", "얇", "떫"),
    "ㄽ": ("곬",),
    "ㄾ": ("핥", "훑"),
    "ㄿ": ("읊",),
    "ㅀ": ("잃", "싫", "닳", "옳", "끓"),
    "ㅁ": ("감", "곰", "꿈", "밤", "솜"),
    "ㅂ": ("밥", "집", "입", "컵", "법"),
    "ㅄ": ("값", "없"),
    "ㅅ": ("갓", "옷", "맛", "빗", "붓"),
    "ㅆ": ("있", "했", "갔", "났", "봤"),
    "ㅇ": ("강", "공", "방", "종", "창"),
    "ㅈ": ("낮", "빚", "젖", "잦", "맺"),
    "ㅊ": ("꽃", "빛", "숯", "낯", "쫓"),
    "ㅋ": ("녘", "엌"),
    "ㅌ": ("밭", "끝", "솥", "겉", "팥"),
    "ㅍ": ("앞", "잎", "옆", "숲", "늪"),
    "ㅎ": ("낳", "놓", "쌓", "좋", "찧"),
}

_FINAL_DELETE_FALLBACK_BASES = ("가", "너", "도", "무", "비", "소", "주", "처")

_FINAL_DELETE_DEFAULT_SOURCES = (
    "각",
    "눈",
    "달",
    "밤",
    "공",
    "옷",
    "집",
    "꽃",
    "밭",
    "앞",
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
                    slot=slot,
                )
                if candidate is None:
                    return None
                if request.trainingType == "SYLLABLE_DELETE":
                    signature = f"source:{candidate['source']}"
                elif request.trainingType in {
                    "FINAL_CONSONANT_CHOICE",
                    "WORD_FINAL_SOUND_CHOICE",
                    "FINAL_CONSONANT_COMPARISON",
                }:
                    signature = f"audioText:{candidate['audioText']}"
                else:
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
        if request.trainingType in {"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"}:
            return [targets[slot]] if slot < len(targets) else []
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
        *,
        slot: int,
    ) -> dict[str, object] | None:
        training_type = request.trainingType
        if training_type == "FINAL_CONSONANT_DELETE":
            return self._final_consonant_delete(request, variant)
        if training_type == "FINAL_CONSONANT_CHOICE":
            return self._isolated_final_sound_choice(request, variant)
        if training_type == "FINAL_CONSONANT_COMPARISON":
            return self._final_consonant_comparison(request, variant)
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
            answer = representative_final_sound(
                correct.surface,
                pronunciation=correct.pronunciation,
            )
            if answer is None:
                return None
            distractors = [
                sound for sound in REPRESENTATIVE_FINAL_SOUNDS if sound != answer
            ]
            rotated_distractors = self._rotate(distractors, variant)
            choices = self._rotate(
                [answer, rotated_distractors[0], rotated_distractors[1]],
                variant,
            )
            return {
                "audioText": correct.spoken_text,
                "choices": choices,
                "answerIndex": choices.index(answer),
            }
        if training_type == "FINAL_CONSONANT_COMPARISON":
            correct_sound = representative_final_sound(
                correct.surface,
                pronunciation=correct.pronunciation,
            )
            if correct_sound is None:
                return None
            options = [
                unit
                for unit in units
                if unit.unit_type == "SYLLABLE"
                and unit.coda
                and unit.onset == correct.onset
                and unit.vowel == correct.vowel
                and representative_final_sound(
                    unit.surface,
                    pronunciation=unit.pronunciation,
                )
                != correct_sound
            ]
            distractors_by_sound: dict[str, LearningUnit] = {}
            for unit in self._rotate(options, variant):
                sound = representative_final_sound(
                    unit.surface,
                    pronunciation=unit.pronunciation,
                )
                if sound is not None and sound not in distractors_by_sound:
                    distractors_by_sound[sound] = unit
            if len(distractors_by_sound) < 2:
                return None
            distractors = list(distractors_by_sound.values())[:2]
            choices = self._rotate(
                [correct.surface, distractors[0].surface, distractors[1].surface],
                variant,
            )
            return {
                "audioText": correct.spoken_text,
                "choices": choices,
                "answerIndex": choices.index(correct.surface),
            }
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
        if training_type == "SYLLABLE_DELETE":
            index = slot % len(correct.surface)
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
                4,
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

    def _final_consonant_delete(
        self,
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object] | None:
        target_codes = [feature.featureCode for feature in request.targetFeatures]
        exact_codas = [
            code.rsplit(".", 1)[-1]
            for code in target_codes
            if code.startswith("GRAPHEME.CODA.")
            and code.rsplit(".", 1)[-1] in _FINAL_DELETE_SOURCES
        ]
        exact_coda = exact_codas[0] if exact_codas else None
        wants_complex = bool(
            {"SYLLABLE.COMPLEX_CODA", "GRAPHEME.CODA.COMPLEX"}.intersection(target_codes)
        )

        if exact_coda:
            sources = [*_FINAL_DELETE_SOURCES.get(exact_coda, ())]
            if len(sources) < request.count:
                sources.extend(
                    self._compose(
                        decompose_text(base)[0].onset,
                        decompose_text(base)[0].nucleus,
                        exact_coda,
                    )
                    for base in _FINAL_DELETE_FALLBACK_BASES
                )
        elif wants_complex:
            sources = [
                source
                for coda, coda_sources in _FINAL_DELETE_SOURCES.items()
                if coda in COMPLEX_CODA_PARTS
                for source in coda_sources
            ]
        else:
            sources = list(_FINAL_DELETE_DEFAULT_SOURCES)

        excluded = set(request.excludedFeatures)
        eligible: list[str] = []
        for source in dict.fromkeys(sources):
            parts = decompose_text(source)
            if len(parts) != 1 or not parts[0].coda:
                continue
            coda = parts[0].coda
            if exact_coda and coda != exact_coda:
                continue
            if wants_complex and coda not in COMPLEX_CODA_PARTS:
                continue
            if f"GRAPHEME.CODA.COMPLEX.{coda}" in excluded:
                continue
            if f"GRAPHEME.CODA.SIMPLE.{coda}" in excluded:
                continue
            if "SYLLABLE.COMPLEX_CODA" in excluded and coda in COMPLEX_CODA_PARTS:
                continue
            eligible.append(source)

        if len(eligible) < request.count:
            return None
        source = eligible[variant % len(eligible)]
        syllable = decompose_text(source)[0]
        result = self._compose(syllable.onset, syllable.nucleus)
        return {
            "source": source,
            "targetAudioText": result,
            "removableUnits": [syllable.onset, syllable.nucleus, syllable.coda],
            "answerIndex": 2,
            "result": result,
        }

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
        sound_text = unit.spoken_text
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

    def _isolated_final_sound_choice(
        self,
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object] | None:
        target_codes = [feature.featureCode for feature in request.targetFeatures]
        exact_coda = next(
            (
                code.rsplit(".", 1)[-1]
                for code in target_codes
                if code.startswith("GRAPHEME.CODA.")
            ),
            None,
        )
        excluded_codas = {
            code.rsplit(".", 1)[-1]
            for code in request.excludedFeatures
            if code.startswith("GRAPHEME.CODA.")
        }
        if exact_coda:
            codas = [exact_coda]
        else:
            codas = [
                coda for coda in _FINAL_CHOICE_WRITTEN_CODAS if coda not in excluded_codas
            ]
        if not codas:
            return None
        coda = codas[variant % len(codas)]
        curated = _FINAL_CHOICE_SYLLABLES.get(coda, ())
        if curated:
            audio_text = curated[(variant // len(codas)) % len(curated)]
        else:
            onset, vowel = _FINAL_CHOICE_BASES[variant % len(_FINAL_CHOICE_BASES)]
            audio_text = self._compose(onset, vowel, coda)
        answer = representative_final_sound(audio_text)
        if answer is None:
            return None
        distractors = [
            sound for sound in REPRESENTATIVE_FINAL_SOUNDS if sound != answer
        ]
        rotated_distractors = self._rotate(distractors, variant)
        choices = self._rotate(
            [answer, rotated_distractors[0], rotated_distractors[1]],
            variant,
        )
        return {
            "audioText": audio_text,
            "choices": choices,
            "answerIndex": choices.index(answer),
        }

    def _final_consonant_comparison(
        self,
        request: TrainingCandidateRequest,
        variant: int,
    ) -> dict[str, object] | None:
        target_codes = [feature.featureCode for feature in request.targetFeatures]
        exact_coda = next(
            (
                code.rsplit(".", 1)[-1]
                for code in target_codes
                if code.startswith("GRAPHEME.CODA.")
            ),
            None,
        )
        excluded_codas = {
            code.rsplit(".", 1)[-1]
            for code in request.excludedFeatures
            if code.startswith("GRAPHEME.CODA.")
        }
        codas = [exact_coda] if exact_coda else [
            coda for coda in _FINAL_CHOICE_WRITTEN_CODAS if coda not in excluded_codas
        ]
        if not codas:
            return None
        coda = codas[variant % len(codas)]
        curated = _FINAL_CHOICE_SYLLABLES.get(coda, ())
        if curated:
            audio_text = curated[(variant // len(codas)) % len(curated)]
            syllable = decompose_text(audio_text)[0]
            onset, vowel = syllable.onset, syllable.nucleus
        else:
            onset, vowel = _FINAL_CHOICE_BASES[variant % len(_FINAL_CHOICE_BASES)]
            audio_text = self._compose(onset, vowel, coda)
        answer_sound = representative_final_sound(audio_text)
        if answer_sound is None:
            return None
        distractor_sounds = [
            sound for sound in REPRESENTATIVE_FINAL_SOUNDS if sound != answer_sound
        ]
        distractor_sounds = self._rotate(distractor_sounds, variant)
        choices = self._rotate(
            [
                audio_text,
                self._compose(onset, vowel, distractor_sounds[0]),
                self._compose(onset, vowel, distractor_sounds[1]),
            ],
            variant,
        )
        return {
            "audioText": audio_text,
            "choices": choices,
            "answerIndex": choices.index(audio_text),
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
