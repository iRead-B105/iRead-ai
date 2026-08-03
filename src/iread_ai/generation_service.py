"""Validated GMS generation with deterministic child-safe fallback content."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .generation_models import (
    TrainingCandidateRequest,
    TrainingCandidateResponse,
)
from .lexicon.contracts import LexiconPaletteRequest, LexiconTargetFeature
from .lexicon.service import LexiconPaletteService, LexiconUnavailableError
from .mock_generators import generate_training_candidates as mock_training_candidates
from .providers import GenerationProviderError, GMSTextProvider
from .training_bank import default_basic_training_generator
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
        "각각 ㅇ·ㄴ·ㅁ으로 발음되는 배열. 예: 국물, 앞니, 밥 먹기, 읽는."
    ),
    "SYLLABLE.COMPLEX_CODA": (
        "겹받침: 표기에 ㄳ·ㄵ·ㄶ·ㄺ·ㄻ·ㄼ·ㄽ·ㄾ·ㄿ·ㅀ·ㅄ 중 하나가 "
        "종성으로 들어간 음절. 예: 몫, 앉다, 읽다, 닮다, 넓다, 없다."
    ),
}

_HYBRID_TYPE_GUIDES = {
    "DIFFICULT_WORD_PREVIEW": (
        "아이 수준에 맞는 자연스러운 한 문장을 만들고, 문장에 실제로 포함된 어려운 "
        "낱말을 골라 한글 음절 단위로 정확히 분해하세요."
    ),
    "SENTENCE_READING": (
        "한 가지 분명한 사건을 담은 자연스러운 문장을 만들고 tokens에는 문장을 읽는 "
        "순서대로 어절을 넣으세요."
    ),
    "SHORT_PASSAGE_READING": (
        "같은 사건이 인과관계로 이어지는 2~3문장의 짧은 글을 만드세요. 각 문장은 "
        "초등학생이 소리 내어 읽기 자연스러워야 합니다."
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
        "사용하세요. 한국어 설명이나 다른 값을 넣지 마세요."
    ),
}


@dataclass(frozen=True, slots=True)
class ProviderResult:
    value: Any
    provider: str
    fallback: bool


def enrich_training_request_with_lexicon(
    request: TrainingCandidateRequest,
    lexicon_service: LexiconPaletteService | None,
) -> TrainingCandidateRequest:
    if (
        not request.useLexicon
        or request.recommendedWords
        or lexicon_service is None
        or request.trainingType not in HYBRID_LLM_TYPES
    ):
        return request
    target_codes = [feature.featureCode for feature in request.targetFeatures]
    max_batchim_ratio = (
        1.0
        if any("CODA" in code or "BATCHIM" in code for code in target_codes)
        else {1: 0.0, 2: 0.34, 3: 0.5, 4: 0.67, 5: 1.0}[request.difficulty]
    )
    try:
        palette = lexicon_service.build_palette(
            LexiconPaletteRequest(
                requestId=f"{request.requestId}-lexicon",
                targetFeatures=[
                    LexiconTargetFeature(
                        featureCode=feature.featureCode,
                        weaknessScore=feature.weaknessScore,
                        confidence=feature.confidence,
                    )
                    for feature in request.targetFeatures
                ],
                excludedFeatures=request.excludedFeatures,
                minSyllables=1,
                maxSyllables=min(request.difficulty + 1, 5),
                maxBatchimRatio=max_batchim_ratio,
                strictPronunciation=True,
                requireTarget=bool(request.targetFeatures),
                includeInflections=False,
                limit=12,
            )
        )
    except LexiconUnavailableError:
        return request
    words = [item.surface for item in palette.items[:12]]
    if not words:
        return request
    return request.model_copy(update={"recommendedWords": words})


def generate_training(
    request: TrainingCandidateRequest,
    provider: GMSTextProvider | None,
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
        return ProviderResult(rule_based, "rule-db", False)
    if provider is None:
        return ProviderResult(
            mock_training_candidates(request),
            "curated-fallback",
            False,
        )

    def generate() -> TrainingCandidateResponse:
        document = provider.generate_json(
            schema_name="iread_training_candidates",
            schema=_training_response_schema(request),
            system_prompt=(
                "당신은 한국어 아동 문해 훈련 문항 생성기입니다. 응답은 JSON Schema만 "
                "따르며, 아동에게 안전하고 공포스럽지 않은 표현을 사용합니다. "
                "targetFeatures를 우선 연습하고 excludedFeatures는 포함하지 않습니다. "
                "정답과 선택지는 서로 모순되거나 중복되면 안 됩니다."
            ),
            user_prompt=json.dumps(
                _training_prompt_document(request),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        result = TrainingCandidateResponse.model_validate(document)
        if result.type != request.trainingType or len(result.data) != request.count:
            raise ValueError("training response type or count did not match request")
        _normalize_mechanical_fields(request, result)
        _validate_output_template(request, result)
        _validate_hybrid_semantics(request, result)
        _reject_unsafe(result.model_dump_json())
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
            return generate()

    return _with_fallback(
        generate_with_one_validation_retry,
        lambda: mock_training_candidates(request),
        f"gms:{provider.model}",
    )


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
    document["generationRules"] = [
        "data의 모든 문항 각각에 모든 targetFeatures를 최소 한 번씩 포함하세요.",
        "featureCode 문자열을 문장에 그대로 쓰지 말고 실제 한국어 예시로 구현하세요.",
        "다섯 문항은 서로 다른 자연스러운 문장으로 만드세요.",
    ]
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


def _with_fallback(
    generate: Callable[[], Any],
    fallback: Callable[[], Any],
    provider_name: str,
) -> ProviderResult:
    try:
        return ProviderResult(generate(), provider_name, False)
    except (GenerationProviderError, ValidationError, TypeError, ValueError) as exception:
        logger.warning(
            "Training generation provider failed; using safe fallback: %s: %s",
            type(exception).__name__,
            exception,
        )
        return ProviderResult(fallback(), "curated-fallback", True)


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
                for entry in words
            ):
                raise ValueError("difficult word preview was inconsistent")
        elif training_type == "SENTENCE_READING":
            sentence = _compact(str(item.get("sentence", "")))
            tokens = _compact("".join(item.get("tokens", [])))
            if sentence != tokens:
                raise ValueError("sentence tokens did not reconstruct the sentence")
        elif training_type == "SHORT_PASSAGE_READING":
            if not 2 <= len(item.get("sentences", [])) <= 3:
                raise ValueError("short passage must contain two or three sentences")
        elif training_type == "SENTENCE_ASSEMBLY":
            cards = item.get("cards", [])
            order = item.get("answerOrder", [])
            if sorted(order) != list(range(len(cards))):
                raise ValueError("answerOrder was not a card permutation")
            if len(cards) > 1 and order == list(range(len(cards))):
                raise ValueError("sentence cards were not shuffled")
            rebuilt = _compact(" ".join(cards[index] for index in order))
            if rebuilt != _compact(str(item.get("completedSentence", ""))):
                raise ValueError("sentence cards did not reconstruct the sentence")
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
            if choices and answer_index >= 0:
                completed = sentence.replace("{{blank}}", str(choices[answer_index]))
                if _compact(completed) != _compact(str(item.get("completedSentence", ""))):
                    raise ValueError("blank answer did not reconstruct the sentence")
                answers = [*choices, *item.get("acceptedAnswers", [])]
                if any(not _particle_agrees(sentence, str(answer)) for answer in answers):
                    raise ValueError("blank answer did not agree with its Korean particle")
        elif training_type == "IMAGE_SENTENCE_MATCH":
            if len(item.get("choices", [])) != 3:
                raise ValueError("image sentence match must contain three choices")
        elif training_type == "SENTENCE_REPEAT":
            if item.get("emotion") not in {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "EXCITED", "CALM"}:
                raise ValueError("sentence emotion was invalid")
        elif training_type == "PHRASE_READING":
            if _compact("".join(item.get("phrases", []))) != _compact(
                str(item.get("sentence", ""))
            ):
                raise ValueError("phrases did not reconstruct the sentence")
        elif training_type == "REPEATED_SENTENCE_READING":
            if not 2 <= int(item.get("repeatCount", 0)) <= 4:
                raise ValueError("repeatCount was outside the supported range")
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


def _validate_answer_index(item: dict[str, Any]) -> None:
    if "answerIndex" not in item:
        return
    values = item.get("choices", item.get("removableUnits", []))
    index = item.get("answerIndex")
    if not isinstance(index, int) or index < 0 or index >= len(values):
        raise ValueError("answerIndex was outside the choice array")


def _normalize_mechanical_fields(
    request: TrainingCandidateRequest,
    response: TrainingCandidateResponse,
) -> None:
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
