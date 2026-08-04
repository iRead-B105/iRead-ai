from __future__ import annotations

from dataclasses import dataclass

from .devtools.training_review_catalog import (
    TRAINING_REVIEW_CATALOG,
    TrainingReviewSpec,
    output_template,
)
from .generation_models import (
    GeneratedTrainingActivity,
    TrainingActivityRequest,
    TrainingActivityResponse,
    TrainingCandidateRequest,
    TrainingCurriculumArea,
    TrainingSetRequest,
    TrainingSetResponse,
)
from .generation_service import enrich_training_request_with_lexicon, generate_training
from .lexicon.service import LexiconPaletteService
from .personalization.analyzer import KoreanReadingAnalyzer
from .providers import GMSTextProvider
from .training_personalization import candidate_fit_rank, select_training_candidate

_SPEC_BY_TYPE = {spec.training_type: spec for spec in TRAINING_REVIEW_CATALOG}

AREA_POOLS: dict[TrainingCurriculumArea, tuple[str, ...]] = {
    "AUTO": (),
    "LETTER_SOUND": tuple(spec.training_type for spec in TRAINING_REVIEW_CATALOG[:13]),
    "BLENDING": tuple(spec.training_type for spec in TRAINING_REVIEW_CATALOG[13:21]),
    "WORD_READING": tuple(spec.training_type for spec in TRAINING_REVIEW_CATALOG[21:26]),
    "SENTENCE": (
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
    ),
    "FLUENCY": tuple(spec.training_type for spec in TRAINING_REVIEW_CATALOG[29:34]),
}

_FAMILY_PATHS: dict[str, tuple[str, ...]] = {
    "VOWEL": (
        "VOWEL_TRACE",
        "VOWEL_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "PHONEME_BLEND",
        "BASIC_SYLLABLE_BUILD",
        "SYLLABLE_BLEND",
        "WORD_READING",
        "NONWORD_READING",
        "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    ),
    "ONSET": (
        "CONSONANT_TRACE",
        "CONSONANT_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "SYLLABLE_INITIAL_CHOICE",
        "WORD_INITIAL_CHOICE",
        "SAME_INITIAL_WORD_CHOICE",
        "SIMILAR_SOUND_CHOICE",
        "PHONEME_BLEND",
        "BASIC_SYLLABLE_BUILD",
        "SYLLABLE_BLEND",
        "WORD_READING",
        "NONWORD_READING",
        "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    ),
    "CODA": (
        "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
        "FINAL_CONSONANT_COMPARISON",
        "PHONEME_BLEND",
        "FINAL_SYLLABLE_BUILD",
        "DOUBLE_FINAL_BUILD",
        "FINAL_CONSONANT_DELETE",
        "WORD_READING",
        "NONWORD_READING",
        "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "FILL_IN_THE_BLANK",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    ),
    "SYLLABLE": (
        "SYLLABLE_TRACE",
        "SYLLABLE_INITIAL_CHOICE",
        "PHONEME_BLEND",
        "BASIC_SYLLABLE_BUILD",
        "FINAL_SYLLABLE_BUILD",
        "DOUBLE_FINAL_BUILD",
        "SYLLABLE_DELETE",
        "SYLLABLE_REPLACE",
        "SYLLABLE_BLEND",
        "WORD_READING",
        "NONWORD_READING",
        "WORD_CHAIN_READING",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
    ),
    "WORD": (
        "WORD_READING",
        "NONWORD_READING",
        "DIFFICULT_WORD_PREVIEW",
        "WORD_CHAIN_READING",
        "SYLLABLE_BLEND",
        "SYLLABLE_DELETE",
        "SYLLABLE_REPLACE",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    ),
    "LANGUAGE": (
        "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SENTENCE_REPEAT",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    ),
}

_VOWEL_SOUNDS = {
    "ㅏ": "아",
    "ㅑ": "야",
    "ㅓ": "어",
    "ㅕ": "여",
    "ㅗ": "오",
    "ㅛ": "요",
    "ㅜ": "우",
    "ㅠ": "유",
    "ㅡ": "으",
    "ㅣ": "이",
    "ㅐ": "애",
    "ㅔ": "에",
    "ㅒ": "얘",
    "ㅖ": "예",
    "ㅘ": "와",
    "ㅙ": "왜",
    "ㅚ": "외",
    "ㅝ": "워",
    "ㅞ": "웨",
    "ㅟ": "위",
    "ㅢ": "의",
}

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


@dataclass(frozen=True, slots=True)
class GeneratedSetResult:
    response: TrainingSetResponse
    providers: tuple[str, ...]


def generate_training_set(
    request: TrainingSetRequest,
    provider: GMSTextProvider | None,
    lexicon_service: LexiconPaletteService | None = None,
    analyzer: KoreanReadingAnalyzer | None = None,
) -> GeneratedSetResult:
    area = _resolved_area(request.curriculumArea, request.difficulty)
    training_types = plan_training_types(request, area)
    activities: list[GeneratedTrainingActivity] = []
    providers: list[str] = []
    for sequence, training_type in enumerate(training_types, start=1):
        activity_request = TrainingActivityRequest(
            requestId=f"{request.requestId}-activity-{sequence}",
            schemaVersion=request.schemaVersion,
            sequence=sequence,
            trainingType=training_type,
            difficulty=request.difficulty,
            targetFeatures=request.targetFeatures,
            excludedFeatures=request.excludedFeatures,
            additionalPrompt=request.additionalPrompt,
            useLexicon=request.useLexicon,
        )
        result = generate_training_activity(
            activity_request,
            provider,
            lexicon_service=lexicon_service,
            analyzer=analyzer,
        )
        activities.append(result.activity)
        providers.append(result.activity.provider)
    return GeneratedSetResult(
        TrainingSetResponse(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            curriculumArea=area,
            focusFeatureCodes=[feature.featureCode for feature in request.targetFeatures],
            activities=activities,
        ),
        tuple(providers),
    )


def generate_training_activity(
    request: TrainingActivityRequest,
    provider: GMSTextProvider | None,
    lexicon_service: LexiconPaletteService | None = None,
    analyzer: KoreanReadingAnalyzer | None = None,
) -> TrainingActivityResponse:
    spec = _required_spec(request.trainingType)
    candidate_request = enrich_training_request_with_lexicon(
        TrainingCandidateRequest(
            requestId=request.requestId,
            schemaVersion=request.schemaVersion,
            trainingType=request.trainingType,
            count=5,
            difficulty=request.difficulty,
            targetFeatures=request.targetFeatures,
            excludedFeatures=request.excludedFeatures,
            additionalPrompt=request.additionalPrompt,
            outputTemplate=output_template(request.trainingType),
            useLexicon=request.useLexicon,
        ),
        lexicon_service,
    )
    recommended_words = candidate_request.recommendedWords
    generated = generate_training(candidate_request, provider)
    item, personalization = select_training_candidate(
        list(generated.value.data),
        target_features=(feature.featureCode for feature in request.targetFeatures),
        excluded_features=request.excludedFeatures,
        recommended_words=recommended_words,
        analyzer=analyzer,
        lexicon_applied=bool(recommended_words),
        training_type=request.trainingType,
        difficulty=request.difficulty,
    )
    selected_fit = personalization.candidates[personalization.selectedCandidateIndex]
    if (
        generated.provider.startswith("gms:")
        and selected_fit.lengthStatus in {"TOO_SHORT", "TOO_LONG"}
    ):
        retry_request = candidate_request.model_copy(
            update={
                "requestId": f"{candidate_request.requestId}-length-retry",
                "additionalPrompt": (
                    candidate_request.additionalPrompt
                    + "\n이전 후보는 읽기 길이를 지키지 못했습니다. "
                    "각 문장을 짧은 한 가지 사건으로 쓰고 readingLengthPolicy의 "
                    "문장별·전체 음절 범위를 반드시 지키세요. 목표 특징은 2~6회, "
                    "추천 어휘는 최대 4개만 자연스럽게 사용하세요."
                ).strip(),
            }
        )
        retry_generated = generate_training(retry_request, provider)
        retry_item, retry_personalization = select_training_candidate(
            list(retry_generated.value.data),
            target_features=(feature.featureCode for feature in request.targetFeatures),
            excluded_features=request.excludedFeatures,
            recommended_words=recommended_words,
            analyzer=analyzer,
            lexicon_applied=bool(recommended_words),
            training_type=request.trainingType,
            difficulty=request.difficulty,
        )
        retry_fit = retry_personalization.candidates[
            retry_personalization.selectedCandidateIndex
        ]
        if candidate_fit_rank(retry_fit) > candidate_fit_rank(selected_fit):
            generated = retry_generated
            item = retry_item
            personalization = retry_personalization
        personalization = personalization.model_copy(update={"generationAttempts": 2})
    _normalize_canonical_audio(request.trainingType, item)
    target_codes = [feature.featureCode for feature in request.targetFeatures]
    activity = GeneratedTrainingActivity(
        sequence=request.sequence,
        templateId=spec.template_id,
        trainingType=spec.training_type,
        name=spec.name,
        group=spec.group,
        strategy=spec.strategy,
        provider=generated.provider,
        targetFeatureCodes=target_codes,
        rationale=_rationale(spec, target_codes),
        item=item,
        personalization=personalization,
    )
    return TrainingActivityResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        activity=activity,
    )


def plan_training_types(
    request: TrainingSetRequest,
    area: TrainingCurriculumArea | None = None,
) -> tuple[str, ...]:
    resolved_area = area or _resolved_area(request.curriculumArea, request.difficulty)
    for training_type in request.preferredTrainingTypes:
        _required_spec(training_type)
    family = _feature_family(
        request.targetFeatures[0].featureCode if request.targetFeatures else ""
    )
    family_path = _FAMILY_PATHS.get(
        family,
        tuple(spec.training_type for spec in TRAINING_REVIEW_CATALOG),
    )
    area_pool = AREA_POOLS[resolved_area]
    ordered = [*request.preferredTrainingTypes]
    ordered.extend(
        training_type
        for training_type in area_pool
        if training_type in family_path and training_type not in ordered
    )
    ordered.extend(training_type for training_type in family_path if training_type not in ordered)
    ordered.extend(training_type for training_type in area_pool if training_type not in ordered)
    ordered.extend(
        spec.training_type for spec in TRAINING_REVIEW_CATALOG if spec.training_type not in ordered
    )
    return tuple(ordered[: request.activityCount])


def _resolved_area(
    area: TrainingCurriculumArea,
    difficulty: int,
) -> TrainingCurriculumArea:
    if area != "AUTO":
        return area
    return {
        1: "LETTER_SOUND",
        2: "BLENDING",
        3: "WORD_READING",
        4: "SENTENCE",
        5: "FLUENCY",
    }[difficulty]


def _feature_family(feature_code: str) -> str:
    if "VOWEL" in feature_code:
        return "VOWEL"
    if "ONSET" in feature_code:
        return "ONSET"
    if "CODA" in feature_code:
        return "CODA"
    if feature_code.startswith("WORD."):
        return "WORD"
    if feature_code.startswith("SYLLABLE."):
        return "SYLLABLE"
    if feature_code.startswith("PHONOLOGY.") or feature_code.startswith("SENTENCE."):
        return "LANGUAGE"
    return "GENERAL"


def _normalize_canonical_audio(training_type: str, item: dict[str, object]) -> None:
    if training_type == "CONSONANT_VOWEL_CLASSIFICATION":
        audio_text = str(item.get("audioText", ""))
        for value, spoken in (*_VOWEL_SOUNDS.items(), *_CONSONANT_NAMES.items()):
            if value in audio_text:
                item["audioText"] = spoken
                return
    if training_type not in {
        "VOWEL_TRACE",
        "CONSONANT_TRACE",
        "SYLLABLE_TRACE",
    }:
        return
    target = str(item.get("target", ""))
    if training_type == "VOWEL_TRACE":
        item["soundText"] = _VOWEL_SOUNDS.get(target, target)
    elif training_type == "CONSONANT_TRACE":
        item["soundText"] = _CONSONANT_NAMES.get(target, target)
    else:
        item["soundText"] = target


def _required_spec(training_type: str) -> TrainingReviewSpec:
    spec = _SPEC_BY_TYPE.get(training_type)
    if spec is None:
        raise ValueError(f"unsupported training type: {training_type}")
    return spec


def _rationale(spec: TrainingReviewSpec, target_codes: list[str]) -> str:
    if target_codes:
        return f"{', '.join(target_codes)} 목표를 {spec.name} 활동으로 연습합니다."
    return f"현재 읽기 수준에 맞춰 {spec.name} 활동을 연습합니다."


__all__ = [
    "AREA_POOLS",
    "GeneratedSetResult",
    "generate_training_activity",
    "generate_training_set",
    "plan_training_types",
]
