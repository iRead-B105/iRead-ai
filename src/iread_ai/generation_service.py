"""Validated LLM training generation with a rule-db item bank and mock-mode curated content.

Real-provider failures propagate as GenerationProviderError so endpoints can
answer 502/503 instead of silently substituting curated items.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .generation_models import (
    TrainingCandidateRequest,
    TrainingCandidateResponse,
    TrainingGenerationMetadata,
    TrainingTargetFeature,
)
from .lexicon.contracts import LexiconPaletteRequest, LexiconTargetFeature
from .lexicon.service import LexiconPaletteService, LexiconUnavailableError
from .mock_generators import generate_training_candidates as mock_training_candidates
from .providers import GenerationProviderError, GMSTextProvider
from .training_bank import default_basic_training_generator
from .training_feature_compatibility import compatible_features
from .training_final_sounds import (
    REPRESENTATIVE_FINAL_SOUNDS,
    representative_final_sound,
)
from .training_language_quality import (
    validate_complete_korean_sentence,
    validate_image_sentence_answer,
)
from .training_length_policy import training_length_policy

logger = logging.getLogger(__name__)

HYBRID_LLM_TYPES = frozenset(
    {
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
    }
)

REAL_WORD_ONLY_TYPES = HYBRID_LLM_TYPES

SEMANTIC_REVIEW_TYPES = frozenset(
    {
        "SHORT_PASSAGE_READING",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
        "SHORT_STORY_READING",
    }
)

LEXICON_RULE_TYPES = frozenset(
    {
        "WORD_INITIAL_CHOICE",
        "SAME_INITIAL_WORD_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
        "SYLLABLE_BLEND",
        "SYLLABLE_DELETE",
        "WORD_READING",
        "NONWORD_READING",
        "WORD_CHAIN_READING",
    }
)

_DEFAULT_PALETTE_KEY = "__DEFAULT__"
_TRAINING_WORD_BLOCKLIST = frozenset({"새끼"})

_UNSAFE_TERMS = (
    "자살",
    "죽여",
    "살해",
    "피투성이",
    "성관계",
    "마약",
    "담배",
    "술을 마",
)

_FEATURE_GUIDES = {
    "PHONOLOGY.NASALIZATION": (
        "비음화: 받침의 대표음 ㄱ·ㄷ·ㅂ 뒤에 초성 ㄴ·ㅁ이 와서 "
        "각각 ㅇ·ㄴ·ㅁ으로 발음되는 배열. 한 낱말 예: 국물, 앞니, 막내, 읽는."
    ),
    "SYLLABLE.COMPLEX_CODA": (
        "겹받침: 표기에 ㄳ·ㄵ·ㄶ·ㄺ·ㄻ·ㄼ·ㄽ·ㄾ·ㄿ·ㅀ·ㅄ 중 하나가 "
        "종성으로 들어간 음절. 예: 몫, 앉다, 읽다, 닮다, 넓다, 없다."
    ),
}

_HYBRID_TYPE_GUIDES = {
    "DIFFICULT_WORD_PREVIEW": (
        "아이 수준에 맞는 자연스러운 한 문장을 만들고, 문장에 실제로 포함된 어려운 "
        "낱말을 골라 한글 음절 단위로 정확히 분해하세요. difficultWords의 word에는 "
        "공백이나 문장부호가 없는 한 개의 한국어 낱말만 넣으세요. '밥 먹기'처럼 "
        "띄어 쓴 구는 금지하며, 비음화 목표라면 '국물' 또는 '앞니' 같은 한 낱말을 쓰세요."
    ),
    "SENTENCE_READING": (
        "한 가지 분명한 사건을 담은 자연스러운 문장을 만들고 tokens에는 문장을 읽는 "
        "순서대로 어절을 넣으세요."
    ),
    "SHORT_PASSAGE_READING": (
        "같은 사건이 인과관계로 이어지는 2~3문장의 짧은 글을 만드세요. 각 문장은 "
        "초등학생이 소리 내어 읽기 자연스러워야 합니다. 한 후보 안에서 주인공과 "
        "핵심 사건을 바꾸지 말고, 추천 단어를 넣으려고 서로 무관한 문장을 나열하지 마세요."
    ),
    "SENTENCE_ASSEMBLY": (
        "자연스러운 완성 문장을 먼저 만든 뒤 어절 카드로 분해하세요. cards는 섞고, "
        "answerOrder는 원문 순서를 복원하는 0 기반 인덱스여야 합니다."
    ),
    "FILL_IN_THE_BLANK": (
        "문맥만으로 답을 고를 수 있는 문장 하나를 만들고 {{blank}}를 정확히 한 번 "
        "사용하세요. inputType은 CHOICE만 사용하세요. 빈칸에는 명사만 들어가며, 빈칸 "
        "바로 뒤에 조사를 반드시 붙이세요. 오답은 같은 품사이되 문맥에는 맞지 않아야 "
        "합니다. 정답과 모든 오답에 동일하게 맞는 조사만 사용하세요. 예를 들어 "
        "받침 있는 명사 뒤에는 을·은·이·과를, 받침 없는 명사 뒤에는 를·는·가·와를 씁니다."
    ),
    "IMAGE_SENTENCE_MATCH": (
        "한 장면으로 명확히 그릴 수 있는 imagePrompt와 그 장면을 정확히 설명하는 문장 "
        "하나, 핵심 행동이나 대상이 다른 오답 두 개를 만드세요."
    ),
    "SENTENCE_REPEAT": (
        "짧고 말하기 자연스러운 문장을 만들고 내용에 맞는 감정을 NEUTRAL, HAPPY, "
        "SAD, ANGRY, EXCITED, CALM 중 하나로 지정하세요."
    ),
    "PHRASE_READING": (
        "자연스러운 한 문장을 조사와 수식 관계를 깨뜨리지 않는 2~4개의 의미 단위로 "
        "나누세요. phrases를 순서대로 합치면 sentence와 같아야 합니다."
    ),
    "REPEATED_SENTENCE_READING": (
        "반복해 읽기 좋은 자연스러운 한 문장과 2~4 사이의 repeatCount를 만드세요."
    ),
    "SHORT_STORY_READING": (
        "한 사건이 시작되고 변화한 뒤 마무리되는 짧은 이야기로 만드세요. 설명과 짧은 "
        "대사를 섞어 정확히 3~4문장으로 쓰세요. 각 문장의 speaker는 NARRATOR 또는 "
        "CHARACTER만 사용하고, "
        "emotion은 NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, EXCITED, CALM 중 하나만 "
        "사용하세요. 한국어 설명이나 다른 값을 넣지 마세요. CHARACTER 문장은 반드시 "
        "등장인물이 직접 말한 짧은 대사를 큰따옴표로 감싸고, 서술문을 CHARACTER로 "
        "표시하지 마세요."
    ),
}


@dataclass(frozen=True, slots=True)
class ProviderResult:
    value: Any
    provider: str


def enrich_training_request_with_lexicon(
    request: TrainingCandidateRequest,
    lexicon_service: LexiconPaletteService | None,
) -> TrainingCandidateRequest:
    if (
        not request.useLexicon
        or lexicon_service is None
        or request.trainingType not in HYBRID_LLM_TYPES | LEXICON_RULE_TYPES
    ):
        return request
    if request.trainingType in LEXICON_RULE_TYPES:
        palettes = _build_target_word_palettes(request, lexicon_service)
        words = list(
            dict.fromkeys(
                word
                for palette_words in palettes.values()
                for word in palette_words
            )
        )[:40]
        if not words:
            return request
        return request.model_copy(
            update={
                "recommendedWords": words,
                "recommendedWordsByFeature": palettes,
            }
        )

    if request.recommendedWords:
        return request
    palette = _build_word_palette(request, lexicon_service, request.targetFeatures, "lexicon")
    if not palette:
        return request
    return request.model_copy(update={"recommendedWords": palette})


def _build_target_word_palettes(
    request: TrainingCandidateRequest,
    lexicon_service: LexiconPaletteService,
) -> dict[str, list[str]]:
    targets = compatible_features(request.trainingType, request.targetFeatures)
    if not targets:
        words = _build_word_palette(request, lexicon_service, [], "lexicon-default")
        return {_DEFAULT_PALETTE_KEY: words} if words else {}

    result: dict[str, list[str]] = {}
    for index, target in enumerate(targets):
        words = _build_word_palette(
            request,
            lexicon_service,
            [target],
            f"lexicon-target-{index + 1}",
        )
        if words:
            result[target.featureCode] = words
    return result


def _build_word_palette(
    request: TrainingCandidateRequest,
    lexicon_service: LexiconPaletteService,
    targets: list[TrainingTargetFeature],
    request_suffix: str,
) -> list[str]:
    target_codes = [feature.featureCode for feature in targets]
    exact_syllable_count = _requested_word_syllable_count(target_codes)
    lexicon_targets = [
        feature
        for feature in targets
        if feature.featureCode.startswith(("GRAPHEME.", "PHONOLOGY.", "PHONO_"))
        or feature.featureCode == "SYLLABLE.COMPLEX_CODA"
    ]
    max_batchim_ratio = (
        1.0
        if any("CODA" in code or "BATCHIM" in code for code in target_codes)
        else {1: 0.0, 2: 0.34, 3: 0.5, 4: 0.67, 5: 1.0}[request.difficulty]
    )
    try:
        palette = lexicon_service.build_palette(
            LexiconPaletteRequest(
                requestId=f"{request.requestId}-{request_suffix}",
                targetFeatures=[
                    LexiconTargetFeature(
                        featureCode=feature.featureCode,
                        weaknessScore=feature.weaknessScore,
                        confidence=feature.confidence,
                    )
                    for feature in lexicon_targets
                ],
                excludedFeatures=request.excludedFeatures,
                partsOfSpeech=["명사"] if request.trainingType in LEXICON_RULE_TYPES else [],
                minSyllables=exact_syllable_count or 1,
                maxSyllables=exact_syllable_count or min(request.difficulty + 1, 5),
                maxBatchimRatio=max_batchim_ratio,
                strictPronunciation=True,
                requireTarget=bool(lexicon_targets),
                includeInflections=False,
                limit=20,
            )
        )
    except LexiconUnavailableError:
        return []
    words: list[str] = []
    for item in palette.items:
        word = item.surface.strip()
        if not re.fullmatch(r"[가-힣]+", word):
            continue
        if word in _TRAINING_WORD_BLOCKLIST:
            continue
        if not _word_matches_training_position(request.trainingType, word, target_codes):
            continue
        if request.trainingType == "WORD_FINAL_SOUND_CHOICE" and (
            item.pronunciationStatus != "NO_CHANGE"
        ):
            continue
        if word not in words:
            words.append(word)
        if len(words) == 20:
            break
    return words


def _requested_word_syllable_count(target_codes: list[str]) -> int | None:
    for code in target_codes:
        match = re.fullmatch(r"WORD\.SYLLABLE_COUNT\.([1-5])", code)
        if match:
            return int(match.group(1))
    return None


def _word_matches_training_position(
    training_type: str,
    word: str,
    target_codes: list[str],
) -> bool:
    from .personalization.hangul import decompose_text

    syllables = decompose_text(word)
    if not syllables:
        return False
    for code in target_codes:
        expected = code.rsplit(".", 1)[-1]
        if training_type in {"WORD_INITIAL_CHOICE", "SAME_INITIAL_WORD_CHOICE"}:
            if code.startswith("GRAPHEME.ONSET.") and syllables[0].onset != expected:
                return False
        if training_type == "WORD_FINAL_SOUND_CHOICE":
            if code.startswith("GRAPHEME.CODA.") and syllables[-1].coda != expected:
                return False
    return True


def generate_training(
    request: TrainingCandidateRequest,
    provider: GMSTextProvider | None,
    lexicon_service: LexiconPaletteService | None = None,
) -> ProviderResult:
    try:
        rule_based = default_basic_training_generator().generate(request)
    except (OSError, RuntimeError, sqlite3.Error) as exception:
        logger.exception(
            "Rule-based training item bank failed; delegating to the text provider: %s",
            exception,
        )
        rule_based = None
    if rule_based is not None:
        return _with_generation_metadata(
            request,
            ProviderResult(
                _validated_training_response(request, rule_based.model_dump(mode="json")),
                "rule-db",
            ),
        )
    if provider is None:
        # Mock generation mode: the curated item bank is the intended provider.
        curated = _validated_training_response(
            request,
            mock_training_candidates(request).model_dump(mode="json"),
        )
        _validate_registered_vocabulary(request, curated, lexicon_service)
        return _with_generation_metadata(
            request,
            ProviderResult(
                curated,
                "curated-fallback",
            ),
        )

    def generate(validation_feedback: str | None = None) -> TrainingCandidateResponse:
        prompt_document = _training_prompt_document(request)
        if validation_feedback:
            prompt_document["previousValidationFailure"] = validation_feedback
            prompt_document["generationRules"].append(
                "이전 응답의 검증 실패 원인을 고쳐 완전히 새로운 후보 "
                f"{request.count}개를 만드세요."
            )
        document = provider.generate_json(
            schema_name="iread_training_candidates",
            schema=_training_response_schema(request),
            system_prompt=(
                "당신은 한국어 아동 문해 훈련 문항 생성기입니다. 응답은 JSON Schema만 "
                "따르며, 아동에게 안전하고 공포스럽지 않은 표현을 사용합니다. "
                "targetFeatures를 우선 연습하고 excludedFeatures는 포함하지 않습니다. "
                "정답과 선택지는 서로 모순되거나 중복되면 안 됩니다."
            ),
            user_prompt=json.dumps(prompt_document, ensure_ascii=False, sort_keys=True),
        )
        result = _validated_training_response(request, document)
        if request.trainingType in SEMANTIC_REVIEW_TYPES:
            result = _review_training_response(request, result, provider)
        _validate_registered_vocabulary(request, result, lexicon_service)
        return result

    def generate_with_one_validation_retry() -> TrainingCandidateResponse:
        try:
            return generate()
        except (ValidationError, TypeError, ValueError) as exception:
            logger.warning(
                "Training generation validation failed; retrying once: %s: %s",
                type(exception).__name__,
                exception,
            )
            feedback = f"{type(exception).__name__}: {exception}"
            return generate(feedback)

    # GenerationProviderError from the provider propagates unchanged; local
    # validation failures are wrapped so callers can answer 502/503 without
    # exposing generated content.
    try:
        response = generate_with_one_validation_retry()
    except (ValidationError, TypeError, ValueError) as exception:
        logger.warning(
            "Training generation failed local validation after one retry: %s",
            type(exception).__name__,
        )
        raise GenerationProviderError(
            "training generation output failed local validation after one retry",
            retryable=True,
        ) from exception
    return _with_generation_metadata(
        request,
        ProviderResult(
            response,
            f"{getattr(provider, 'provider_name', 'gms')}:{provider.model}",
        ),
    )


def _validated_training_response(
    request: TrainingCandidateRequest,
    document: dict[str, Any],
) -> TrainingCandidateResponse:
    result = TrainingCandidateResponse.model_validate(document)
    if result.type != request.trainingType or len(result.data) != request.count:
        raise ValueError("training response type or count did not match request")
    _normalize_mechanical_fields(request, result)
    _validate_output_template(request, result)
    _validate_final_sound_choices(request, result)
    _validate_hybrid_semantics(request, result)
    _validate_candidate_uniqueness(result)
    _reject_unsafe(result.model_dump_json())
    return result


def _review_training_response(
    request: TrainingCandidateRequest,
    draft: TrainingCandidateResponse,
    provider: GMSTextProvider,
) -> TrainingCandidateResponse:
    review_document = {
        "task": "아동용 훈련 문항의 한국어 문법과 의미 관계를 검토하고 최소한으로 고치기",
        "rules": [
            "JSON 구조, 문항 수, 목표 특징, 난이도와 정답 위치는 유지하세요.",
            "어색한 조사, 잘못된 활용, 주어와 서술어 호응을 바로잡으세요.",
            "짧은 글과 이야기의 문장들은 같은 사건의 원인, 행동, 결과 순서로 이어지게 하세요.",
            "추천 단어를 모두 넣기 위해 주인공이나 소재를 바꾸지 말고, 문맥을 깨는 단어는 빼세요.",
            "'수아는 국물이 먹고 싶었어요'처럼 목적어에 이/가를 붙이지 말고 "
            "'수아는 국물을 먹고 싶었어요'처럼 조사와 서술어의 관계를 바로잡으세요.",
            "SHORT_STORY_READING의 CHARACTER 문장은 큰따옴표 안의 직접 대사여야 합니다.",
            "그림 문항은 imagePrompt와 정확히 일치하는 선택지가 answerIndex에 오게 하세요.",
            "빈칸 문항은 정답만 문맥상 자연스럽고 오답은 분명히 틀리게 하세요.",
            "원문에 없던 새 인물이나 새 사건을 불필요하게 추가하지 마세요.",
            "문장과 선택지에는 실제 한국어 사전에 등재된 낱말만 사용하세요. "
            "목표 글자를 넣기 위해 '까나' 같은 낯선 낱말을 새로 만들지 마세요.",
            "설명 없이 수정된 JSON 객체만 출력하세요.",
        ],
        "requestPolicy": _training_prompt_document(request),
        "draft": draft.model_dump(mode="json"),
    }
    document = provider.generate_json(
        schema_name="iread_training_candidates_reviewed",
        schema=_training_response_schema(request),
        system_prompt=(
            "당신은 초등 저학년 한국어 교재의 교정자입니다. 초안을 새로 창작하지 말고 "
            "문법, 문맥, 정답 근거가 명확하도록 최소한으로 교정합니다."
        ),
        user_prompt=json.dumps(review_document, ensure_ascii=False, sort_keys=True),
    )
    return _validated_training_response(request, document)


def _training_prompt_document(request: TrainingCandidateRequest) -> dict[str, Any]:
    document = request.model_dump(mode="json")
    target_codes = {feature.featureCode for feature in request.targetFeatures}
    document["targetFeatureGuide"] = [
        {
            "featureCode": feature.featureCode,
            "instruction": _FEATURE_GUIDES.get(
                feature.featureCode,
                f"각 문항에 {feature.featureCode} 특징을 실제 표기 또는 발음 배열로 포함하세요.",
            ),
        }
        for feature in request.targetFeatures
    ]
    target_plan = _target_distribution(request)
    document["targetDistribution"] = [
        {
            "dataIndex": index,
            "featureCodes": [feature.featureCode for feature in features],
        }
        for index, features in enumerate(target_plan)
    ]
    document["generationRules"] = [
        "각 data 문항에는 targetDistribution의 같은 dataIndex에 배정된 목표만 "
        "최소 한 번 포함하세요.",
        "한 문항에 모든 targetFeatures를 억지로 동시에 넣지 마세요.",
        "첫 번째 목표는 3문항, 두 번째 목표는 2문항에 분산하세요.",
        "featureCode 문자열을 문장에 그대로 쓰지 말고 실제 한국어 예시로 구현하세요.",
        "다섯 문항은 서로 다른 자연스러운 문장으로 만드세요.",
    ]
    if request.trainingType in REAL_WORD_ONLY_TYPES:
        document["lexicalPolicy"] = {
            "mode": "REAL_WORD_ONLY",
            "scope": "아동이 읽거나 선택하는 모든 문장과 낱말",
        }
        document["generationRules"].extend(
            (
                "문장·짧은 글 단계에서는 실제 한국어 사전에 등재된 낱말만 사용하세요.",
                "목표 글자나 음운을 넣기 위해 낯선 이름이나 무의미 낱말을 만들지 마세요.",
                "목표를 만족하는 등재어가 떠오르지 않으면 verifiedVocabularyPalette의 "
                "다른 단어로 문장을 다시 구성하세요.",
            )
        )
    else:
        document["lexicalPolicy"] = {
            "mode": "PSEUDOWORD_ALLOWED",
            "scope": "글자·음절·낱말 단위 훈련",
        }
    if request.targetFeatures:
        document["generationRules"].append(
            "목표 특징은 짧은 이야기 전체 2~6회, 한 문장 훈련은 1~2회만 사용하세요. "
            "같은 글자나 낱말을 과도하게 반복하지 마세요."
        )
    if request.useLexicon and request.recommendedWords:
        document["verifiedVocabularyPalette"] = request.recommendedWords
        document["generationRules"].extend(
            (
                "verifiedVocabularyPalette의 단어는 문맥에 자연스럽게 맞을 때만 우선 사용하세요.",
                "팔레트 단어를 억지로 넣거나 기본형 그대로 복사하지 마세요.",
                "동사와 형용사는 문장의 시제와 문법에 맞게 자연스럽게 활용하세요.",
                "한 문항에는 팔레트 단어를 최대 4개만 사용하세요.",
                "팔레트 밖 단어가 필요하면 targetFeatures와 excludedFeatures를 먼저 지키세요.",
            )
        )
    length_policy = training_length_policy(request.trainingType, request.difficulty)
    if length_policy is not None:
        document["readingLengthPolicy"] = {
            "countUnit": "공백과 문장부호를 제외한 한글 음절 수",
            "sentenceMin": length_policy.sentence_min,
            "sentenceMax": length_policy.sentence_max,
            "totalMin": length_policy.total_min,
            "totalMax": length_policy.total_max,
            "shortSentenceExamples": [
                "토끼가 작은 상자를 꺼냈어요.",
                "안에서 종소리가 들렸어요.",
                "친구가 깜짝 놀라 뒤로 갔어요.",
                "둘은 상자를 천천히 열었어요.",
            ],
        }
        document["generationRules"].append(
            "각 읽기 문장은 readingLengthPolicy의 sentenceMin~sentenceMax 범위를 지키세요."
        )
        if length_policy.total_max is not None:
            document["generationRules"].append(
                "전체 읽기 분량도 readingLengthPolicy의 totalMin~totalMax 범위를 지키세요."
            )
    if request.trainingType in _HYBRID_TYPE_GUIDES:
        document["trainingTypeGuide"] = _HYBRID_TYPE_GUIDES[request.trainingType]
    if request.trainingType == "NONWORD_READING":
        document["generationRules"].extend(
            (
                "비단어 읽기에서는 실제 낱말과 발음 가능한 무의미 낱말을 모두 포함하세요.",
                "무의미 낱말은 한국어 초성·중성·종성 결합 규칙을 따르되 "
                "사전에 있는 낱말을 그대로 쓰지 마세요.",
            )
        )
    if request.trainingType == "SHORT_STORY_READING":
        document["storyQualityRules"] = [
            "첫 문장에서 인물이 원하는 것 또는 작은 문제를 제시하세요.",
            "인물이 실제 행동을 하고 그 결과로 장면이 달라져야 합니다.",
            "친구를 만나고 놀았다는 식의 사건 없는 나열로 끝내지 마세요.",
            "추천 단어를 나열하기 위해 부자연스러운 문장을 만들지 마세요.",
        ]
    if {
        "PHONOLOGY.NASALIZATION",
        "SYLLABLE.COMPLEX_CODA",
    }.issubset(target_codes):
        document["combinedFeatureGuide"] = {
            "requirement": (
                "다섯 문장 각각에 비음화와 겹받침이 모두 있어야 합니다. "
                "둘 중 하나만 있는 문장은 허용되지 않습니다."
            ),
            "verifiedExpressions": [
                "책을 읽는 아이",
                "걱정 없는 하루",
                "물건의 값만 적기",
                "닭 먹이를 주기",
                "흙놀이를 하기",
                "내 몫만 챙기기",
            ],
            "usage": (
                "각 문항마다 서로 다른 verifiedExpressions 하나 이상을 자연스럽게 활용하세요. "
                "표현을 그대로 복사할 필요는 없지만 겹받침 표기는 유지하세요."
            ),
        }
    if request.trainingType == "SENTENCE_ASSEMBLY":
        document["trainingTypeGuide"] = {
            "cards": "완성 문장을 어절 또는 의미 단위 카드로 나눈 문자열 배열",
            "answerOrder": (
                "cards를 완성 문장 순서로 배치하는 0 기반 정수 인덱스 배열. "
                "cards가 5개면 0,1,2,3,4를 중복 없이 정확히 한 번씩 사용하세요."
            ),
            "completedSentence": "answerOrder 순서대로 cards를 이어 만든 자연스러운 완성 문장",
        }
    return document


def _target_distribution(
    request: TrainingCandidateRequest,
) -> list[list[TrainingTargetFeature]]:
    targets = request.targetFeatures
    if len(targets) <= 1:
        return [targets for _ in range(request.count)]
    primary_count = (request.count + 1) // 2
    return [
        [targets[0] if index < primary_count else targets[1]]
        for index in range(request.count)
    ]


def _training_response_schema(request: TrainingCandidateRequest) -> dict[str, Any]:
    template_data = request.outputTemplate.get("data")
    example = template_data[0] if isinstance(template_data, list) and template_data else {}
    item_schema = _schema_from_template(example)
    if request.trainingType == "SHORT_STORY_READING":
        sentences_schema = item_schema.get("properties", {}).get("sentences", {})
        sentences_schema["minItems"] = 3
        sentences_schema["maxItems"] = 4
        sentence_properties = (
            sentences_schema
            .get("items", {})
            .get("properties", {})
        )
        if sentence_properties:
            sentence_properties["speaker"] = {
                "type": "string",
                "enum": ["NARRATOR", "CHARACTER"],
            }
            sentence_properties["emotion"] = {
                "type": "string",
                "enum": [
                    "NEUTRAL",
                    "HAPPY",
                    "SAD",
                    "ANGRY",
                    "SURPRISED",
                    "EXCITED",
                    "CALM",
                ],
            }
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": [request.trainingType]},
            "data": {
                "type": "array",
                "items": item_schema,
                "minItems": request.count,
                "maxItems": request.count,
            },
        },
        "required": ["type", "data"],
        "additionalProperties": False,
    }


def _schema_from_template(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        properties = {key: _schema_from_template(item) for key, item in value.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item_schema = _schema_from_template(value[0]) if value else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if value is None:
        return {"type": ["string", "null"]}
    if value == "<integer>":
        return {"type": "integer"}
    if value == "<number>":
        return {"type": "number"}
    if value == "<boolean>":
        return {"type": "boolean"}
    return {"type": "string"}


def _with_generation_metadata(
    request: TrainingCandidateRequest,
    result: ProviderResult,
) -> ProviderResult:
    if result.provider == "rule-db":
        provider = "rule-db"
        model = "korean-training-bank-v1"
        strategy = "RULE_DB"
    elif result.provider.startswith(("gms:", "openai:")):
        provider = result.provider.split(":", 1)[0]
        model = result.provider.split(":", 1)[1]
        strategy = "LLM_WITH_LOCAL_VALIDATION"
    else:
        provider = "curated-fallback"
        model = "curated-training-fallback-v1"
        strategy = "CURATED_FALLBACK"
    metadata = TrainingGenerationMetadata(
        provider=provider,
        model=model,
        strategy=strategy,
        lexicalPolicy=(
            "REAL_WORD_ONLY"
            if request.trainingType in REAL_WORD_ONLY_TYPES
            else "PSEUDOWORD_ALLOWED"
        ),
        lexiconApplied=bool(
            request.useLexicon
            and (request.recommendedWords or request.recommendedWordsByFeature)
        ),
    )
    return ProviderResult(
        result.value.model_copy(update={"generationMetadata": metadata}),
        result.provider,
    )


def _validate_registered_vocabulary(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
    lexicon_service: LexiconPaletteService | None,
) -> None:
    if request.trainingType not in REAL_WORD_ONLY_TYPES or lexicon_service is None:
        return
    try:
        unknown = lexicon_service.unknown_content_words(
            _reading_texts_for_lexicon(request.trainingType, response)
        )
    except LexiconUnavailableError as exception:
        logger.warning("Registered-word validation was unavailable: %s", exception)
        return
    if unknown:
        raise ValueError(
            "sentence-level content contained unregistered Korean words: "
            + ", ".join(unknown)
        )


def _reading_texts_for_lexicon(
    training_type: str,
    response: TrainingCandidateResponse,
) -> list[str]:
    texts: list[str] = []
    for item in response.data:
        if training_type == "DIFFICULT_WORD_PREVIEW":
            texts.append(str(item.get("sentence", "")))
        elif training_type in {
            "SENTENCE_READING",
            "SENTENCE_REPEAT",
            "PHRASE_READING",
            "REPEATED_SENTENCE_READING",
        }:
            texts.append(str(item.get("sentence", "")))
        elif training_type == "SHORT_PASSAGE_READING":
            texts.extend(str(sentence) for sentence in item.get("sentences", []))
        elif training_type == "SENTENCE_ASSEMBLY":
            texts.append(str(item.get("completedSentence", "")))
        elif training_type == "FILL_IN_THE_BLANK":
            texts.append(str(item.get("completedSentence", "")))
            texts.extend(str(choice) for choice in item.get("choices", []))
        elif training_type == "IMAGE_SENTENCE_MATCH":
            texts.extend(str(choice) for choice in item.get("choices", []))
        elif training_type == "SHORT_STORY_READING":
            texts.extend(str(line.get("text", "")) for line in item.get("sentences", []))
    return [text for text in texts if text.strip()]


def _validate_output_template(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
    template_data = request.outputTemplate.get("data")
    if not isinstance(template_data, list) or not template_data:
        return
    example = template_data[0]
    if not isinstance(example, dict):
        return
    required_keys = set(example)
    if required_keys and any(not required_keys.issubset(item) for item in response.data):
        raise ValueError("training candidate did not match outputTemplate keys")


def _validate_hybrid_semantics(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
    if request.trainingType not in HYBRID_LLM_TYPES:
        return
    for item in response.data:
        _validate_answer_index(item)
        training_type = request.trainingType
        if training_type == "DIFFICULT_WORD_PREVIEW":
            sentence = str(item.get("sentence", ""))
            words = item.get("difficultWords", [])
            if not words or any(
                str(entry.get("word", "")) not in sentence
                or "".join(entry.get("syllables", [])) != entry.get("word")
                or not re.fullmatch(r"[가-힣]+", str(entry.get("word", "")))
                or any(
                    not re.fullmatch(r"[가-힣]", str(syllable))
                    for syllable in entry.get("syllables", [])
                )
                for entry in words
            ):
                raise ValueError(
                    "difficultWords must contain one Korean word without spaces, "
                    "must occur in the sentence, and syllables must reconstruct it"
                )
            validate_complete_korean_sentence(sentence)
        elif training_type == "SENTENCE_READING":
            raw_sentence = str(item.get("sentence", ""))
            sentence = _compact(raw_sentence)
            tokens = _compact("".join(item.get("tokens", [])))
            if sentence != tokens:
                raise ValueError("sentence tokens did not reconstruct the sentence")
            validate_complete_korean_sentence(raw_sentence)
        elif training_type == "SHORT_PASSAGE_READING":
            sentences = item.get("sentences", [])
            if not 2 <= len(sentences) <= 3:
                raise ValueError("short passage must contain two or three sentences")
            if len(set(sentences)) != len(sentences):
                raise ValueError("short passage sentences must not be duplicated")
            for sentence in sentences:
                validate_complete_korean_sentence(str(sentence))
        elif training_type == "SENTENCE_ASSEMBLY":
            cards = item.get("cards", [])
            order = item.get("answerOrder", [])
            if not 2 <= len(cards) <= 6 or any(not str(card).strip() for card in cards):
                raise ValueError("sentence assembly must contain two to six non-empty cards")
            if sorted(order) != list(range(len(cards))):
                raise ValueError("answerOrder was not a card permutation")
            if len(cards) > 1 and order == list(range(len(cards))):
                raise ValueError("sentence cards were not shuffled")
            rebuilt = _compact(" ".join(cards[index] for index in order))
            completed = str(item.get("completedSentence", ""))
            if rebuilt != _compact(completed):
                raise ValueError("sentence cards did not reconstruct the sentence")
            validate_complete_korean_sentence(completed)
        elif training_type == "FILL_IN_THE_BLANK":
            sentence = str(item.get("sentence", ""))
            if sentence.count("{{blank}}") != 1:
                raise ValueError("fill-in sentence must contain one blank")
            if item.get("inputType") != "CHOICE":
                raise ValueError("fill-in inputType must be CHOICE")
            if _BLANK_PARTICLE_PATTERN.search(sentence) is None:
                raise ValueError("fill-in blank must be followed by a Korean particle")
            choices = item.get("choices", [])
            answer_index = int(item.get("answerIndex", -1))
            if len(choices) != 3 or len(set(choices)) != len(choices):
                raise ValueError("fill-in choices must contain three unique answers")
            if choices and answer_index >= 0:
                completed = sentence.replace("{{blank}}", str(choices[answer_index]))
                if _compact(completed) != _compact(str(item.get("completedSentence", ""))):
                    raise ValueError("blank answer did not reconstruct the sentence")
                answers = [*choices, *item.get("acceptedAnswers", [])]
                if any(not _particle_agrees(sentence, str(answer)) for answer in answers):
                    raise ValueError("blank answer did not agree with its Korean particle")
                if str(choices[answer_index]) not in item.get("acceptedAnswers", []):
                    raise ValueError("blank correct choice must be an accepted answer")
                validate_complete_korean_sentence(completed)
        elif training_type == "IMAGE_SENTENCE_MATCH":
            choices = item.get("choices", [])
            if len(choices) != 3:
                raise ValueError("image sentence match must contain three choices")
            if len(set(choices)) != len(choices):
                raise ValueError("image sentence choices must be unique")
            for sentence in choices:
                validate_complete_korean_sentence(str(sentence))
            validate_image_sentence_answer(
                str(item.get("imagePrompt", "")),
                [str(choice) for choice in choices],
                int(item.get("answerIndex", -1)),
            )
        elif training_type == "SENTENCE_REPEAT":
            if item.get("emotion") not in {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "EXCITED", "CALM"}:
                raise ValueError("sentence emotion was invalid")
            validate_complete_korean_sentence(str(item.get("sentence", "")))
        elif training_type == "PHRASE_READING":
            sentence = str(item.get("sentence", ""))
            phrases = item.get("phrases", [])
            if not 2 <= len(phrases) <= 4 or any(not str(phrase).strip() for phrase in phrases):
                raise ValueError("phrase reading must contain two to four non-empty phrases")
            if _compact("".join(phrases)) != _compact(sentence):
                raise ValueError("phrases did not reconstruct the sentence")
            validate_complete_korean_sentence(sentence)
        elif training_type == "REPEATED_SENTENCE_READING":
            if not 2 <= int(item.get("repeatCount", 0)) <= 4:
                raise ValueError("repeatCount was outside the supported range")
            validate_complete_korean_sentence(str(item.get("sentence", "")))
        elif training_type == "SHORT_STORY_READING":
            lines = item.get("sentences", [])
            if not 3 <= len(lines) <= 4:
                raise ValueError("short story length was invalid")
            if any(
                line.get("speaker") not in {"NARRATOR", "CHARACTER"}
                or line.get("emotion")
                not in {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISED", "EXCITED", "CALM"}
                for line in lines
            ):
                raise ValueError("short story speaker or emotion was invalid")
            texts = [str(line.get("text", "")) for line in lines]
            if len(set(texts)) != len(texts):
                raise ValueError("short story sentences must not be duplicated")
            character_lines = [
                str(line.get("text", ""))
                for line in lines
                if line.get("speaker") == "CHARACTER"
            ]
            if not character_lines or any(
                re.search(r"[“\"][^”\"]+[.!?]s*[”\"]", text) is None
                for text in character_lines
            ):
                raise ValueError(
                    "short story must contain direct quoted dialogue for CHARACTER lines"
                )
            for text in texts:
                validate_complete_korean_sentence(text)


def _validate_candidate_uniqueness(response: TrainingCandidateResponse) -> None:
    canonical_candidates = {
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in response.data
    }
    if len(canonical_candidates) != len(response.data):
        raise ValueError("training candidates must not be duplicated")


def _validate_answer_index(item: dict[str, Any]) -> None:
    if "answerIndex" not in item:
        return
    values = item.get("choices", item.get("removableUnits", []))
    index = item.get("answerIndex")
    if not isinstance(index, int) or index < 0 or index >= len(values):
        raise ValueError("answerIndex was outside the choice array")


def _validate_final_sound_choices(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
    if request.trainingType == "FINAL_CONSONANT_COMPARISON":
        audio_targets: set[str] = set()
        for item in response.data:
            _validate_answer_index(item)
            audio_text = str(item.get("audioText", "")).strip()
            choices = [str(choice).strip() for choice in item.get("choices", [])]
            if not _is_one_hangul_syllable(audio_text):
                raise ValueError("final consonant comparison audioText must be one syllable")
            if (
                len(choices) != 3
                or len(set(choices)) != 3
                or not all(_is_one_hangul_syllable(choice) for choice in choices)
            ):
                raise ValueError(
                    "final consonant comparison choices must be three unique syllables"
                )
            answer_index = int(item["answerIndex"])
            if choices[answer_index] != audio_text:
                raise ValueError(
                    "final consonant comparison answer must match the heard syllable"
                )
            bases = {(ord(choice) - 0xAC00) // 28 for choice in choices}
            if len(bases) != 1:
                raise ValueError(
                    "final consonant comparison choices must share onset and vowel"
                )
            heard_sounds = [representative_final_sound(choice) for choice in choices]
            if None in heard_sounds or len(set(heard_sounds)) != len(heard_sounds):
                raise ValueError(
                    "final consonant comparison choices must have different heard finals"
                )
            if audio_text in audio_targets:
                raise ValueError(
                    "final consonant comparison candidates must use different audio targets"
                )
            audio_targets.add(audio_text)
        return
    if request.trainingType not in {
        "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
    }:
        return
    audio_targets: set[str] = set()
    allowed = set(REPRESENTATIVE_FINAL_SOUNDS)
    for item in response.data:
        _validate_answer_index(item)
        audio_text = str(item.get("audioText", "")).strip()
        choices = [str(choice) for choice in item.get("choices", [])]
        hangul_syllables = re.findall(r"[가-힣]", audio_text)
        if request.trainingType == "FINAL_CONSONANT_CHOICE":
            if len(hangul_syllables) != 1 or audio_text != hangul_syllables[0]:
                raise ValueError(
                    "final consonant choice audioText must be one Hangul syllable"
                )
        elif not re.fullmatch(r"[가-힣]+", audio_text):
            raise ValueError("word final sound choice audioText must be one Korean word")
        if len(choices) != 3 or len(set(choices)) != 3 or not set(choices) <= allowed:
            raise ValueError(
                "final sound choices must contain three unique representative final sounds"
            )
        expected = representative_final_sound(audio_text)
        answer_index = int(item["answerIndex"])
        if expected is None or choices[answer_index] != expected:
            raise ValueError("final sound answer did not match the standard final sound")
        if audio_text in audio_targets:
            raise ValueError("final sound candidates must use different audio targets")
        audio_targets.add(audio_text)


def _is_one_hangul_syllable(value: str) -> bool:
    return len(value) == 1 and "가" <= value <= "힣"


def _normalize_mechanical_fields(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
    if request.trainingType in {
        "VOWEL_TRACE",
        "CONSONANT_TRACE",
        "SYLLABLE_TRACE",
    }:
        for item in response.data:
            target = str(item.get("target", "")).strip()
            if target:
                item["soundText"] = target
        return
    if request.trainingType != "SENTENCE_ASSEMBLY":
        return
    for item in response.data:
        cards = item.get("cards", [])
        order = item.get("answerOrder", [])
        if len(cards) <= 1 or order != list(range(len(cards))):
            continue
        old_positions = [len(cards) - 1, *range(len(cards) - 1)]
        item["cards"] = [cards[position] for position in old_positions]
        new_position = {old: new for new, old in enumerate(old_positions)}
        item["answerOrder"] = [new_position[position] for position in order]


def _compact(value: str) -> str:
    return "".join(value.split()).rstrip(".?!。？！")


_BLANK_PARTICLE_PATTERN = re.compile(
    r"\{\{blank\}\}\s*(으로|로|은|는|이|가|을|를|과|와)(?=\s|[,.!?]|$)"
)
_PARTICLE_BY_CODA = {
    "은": True,
    "는": False,
    "이": True,
    "가": False,
    "을": True,
    "를": False,
    "과": True,
    "와": False,
}


def _particle_agrees(sentence_template: str, answer: str) -> bool:
    match = _BLANK_PARTICLE_PATTERN.search(sentence_template)
    if match is None:
        return True
    syllables = [character for character in answer.strip() if "가" <= character <= "힣"]
    if not syllables:
        return False
    coda_index = (ord(syllables[-1]) - 0xAC00) % 28
    particle = match.group(1)
    if particle in {"으로", "로"}:
        has_non_rieul_coda = coda_index not in {0, 8}
        return (particle == "으로") == has_non_rieul_coda
    return _PARTICLE_BY_CODA[particle] == (coda_index != 0)


def _reject_unsafe(text: str) -> None:
    if any(term in text for term in _UNSAFE_TERMS):
        raise ValueError("generated content failed the child-safety filter")
