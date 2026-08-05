from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iread_ai.devtools.training_review_catalog import (
    TRAINING_REVIEW_CATALOG,
    TrainingReviewSpec,
    output_template,
)

SERVICE_TEMPLATE_IDS = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
    }
)

SERVICE_TRAINING_CATALOG: tuple[TrainingReviewSpec, ...] = tuple(
    spec for spec in TRAINING_REVIEW_CATALOG if spec.template_id in SERVICE_TEMPLATE_IDS
)

_SPEC_BY_ID = {spec.template_id: spec for spec in SERVICE_TRAINING_CATALOG}
_SPEC_BY_TYPE = {spec.training_type: spec for spec in SERVICE_TRAINING_CATALOG}

CHOICE_TYPES = frozenset(
    {
        "CONSONANT_SOUND_CHOICE",
        "VOWEL_SOUND_CHOICE",
        "SYLLABLE_INITIAL_CHOICE",
        "WORD_INITIAL_CHOICE",
        "SAME_INITIAL_WORD_CHOICE",
        "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE",
        "FINAL_CONSONANT_COMPARISON",
        "SIMILAR_SOUND_CHOICE",
        "FILL_IN_THE_BLANK",
        "IMAGE_SENTENCE_MATCH",
    }
)

TRACE_TYPES = frozenset({"VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE"})

BUILD_TYPES = frozenset(
    {"BASIC_SYLLABLE_BUILD", "FINAL_SYLLABLE_BUILD", "DOUBLE_FINAL_BUILD"}
)

READING_TYPES = frozenset(
    {
        "WORD_READING",
        "NONWORD_READING",
        "SENTENCE_READING",
        "SHORT_PASSAGE_READING",
        "SENTENCE_REPEAT",
        "WORD_CHAIN_READING",
        "PHRASE_READING",
        "REPEATED_SENTENCE_READING",
        "SHORT_STORY_READING",
    }
)


def service_training_spec_by_id(template_id: int) -> TrainingReviewSpec | None:
    return _SPEC_BY_ID.get(template_id)


def service_training_spec_by_type(training_type: str) -> TrainingReviewSpec | None:
    return _SPEC_BY_TYPE.get(training_type)


def service_training_groups() -> dict[str, tuple[TrainingReviewSpec, ...]]:
    grouped: dict[str, list[TrainingReviewSpec]] = {}
    for spec in SERVICE_TRAINING_CATALOG:
        grouped.setdefault(spec.group, []).append(spec)
    return {group: tuple(specs) for group, specs in grouped.items()}


def build_target_features(
    feature_profiles: Sequence[Mapping[str, Any]],
    target_codes: Sequence[str],
) -> list[dict[str, Any]]:
    profiles_by_code = {
        str(profile.get("featureCode", "")): profile for profile in feature_profiles
    }
    targets: list[dict[str, Any]] = []
    for feature_code in target_codes[:2]:
        profile = profiles_by_code.get(feature_code, {})
        targets.append(
            {
                "featureCode": feature_code,
                "weaknessScore": float(profile.get("weaknessScore", 0.65)),
                "confidence": float(profile.get("confidence", 0.7)),
                "evidenceCount": int(profile.get("evidenceCount", 5)),
            }
        )
    return targets


def build_candidate_request(
    *,
    request_id: str,
    training_type: str,
    difficulty: int,
    target_features: Sequence[Mapping[str, Any]],
    excluded_features: Sequence[str] = (),
    additional_prompt: str = "",
    use_lexicon: bool = True,
) -> dict[str, Any]:
    spec = service_training_spec_by_type(training_type)
    if spec is None:
        raise ValueError(f"서비스에 등록되지 않은 훈련 유형입니다: {training_type}")
    return {
        "requestId": request_id,
        "schemaVersion": 2,
        "trainingType": training_type,
        "count": 5,
        "difficulty": difficulty,
        "targetFeatures": [dict(feature) for feature in target_features[:2]],
        "excludedFeatures": list(excluded_features),
        "additionalPrompt": additional_prompt,
        "outputTemplate": output_template(training_type),
        "useLexicon": use_lexicon,
        "recommendedWords": [],
        "recommendedWordsByFeature": {},
    }


def choice_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("label") or value)
    return str(value)


def correct_choice(candidate: Mapping[str, Any]) -> str | None:
    choices = candidate.get("choices")
    answer_index = candidate.get("answerIndex")
    if not isinstance(choices, list) or not isinstance(answer_index, int):
        return None
    if answer_index < 0 or answer_index >= len(choices):
        return None
    return choice_label(choices[answer_index])


def expected_text(training_type: str, candidate: Mapping[str, Any]) -> str:
    if training_type == "SENTENCE_ASSEMBLY":
        return str(candidate.get("completedSentence", ""))
    if training_type in {"SYLLABLE_BLEND", "SYLLABLE_DELETE", "SYLLABLE_REPLACE"}:
        return str(candidate.get("result", ""))
    if training_type == "FINAL_CONSONANT_DELETE":
        return str(candidate.get("result", ""))
    if training_type in BUILD_TYPES:
        return str(candidate.get("result", ""))
    if training_type in TRACE_TYPES:
        return str(candidate.get("target", ""))
    if training_type in CHOICE_TYPES:
        return correct_choice(candidate) or ""
    if "sentence" in candidate:
        return str(candidate.get("sentence", ""))
    if "sentences" in candidate:
        values = candidate.get("sentences", [])
        if isinstance(values, list):
            return " ".join(
                str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
                for value in values
            ).strip()
    if "words" in candidate:
        values = candidate.get("words", [])
        if isinstance(values, list):
            return " ".join(
                str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
                for value in values
            ).strip()
    return str(
        candidate.get("targetAudioText")
        or candidate.get("audioText")
        or candidate.get("target")
        or ""
    )


def prompt_text(training_type: str, candidate: Mapping[str, Any]) -> str:
    if training_type in TRACE_TYPES:
        return "글자 모양을 따라 보고 소리 내어 읽어 보세요."
    if training_type in CHOICE_TYPES:
        if training_type == "FILL_IN_THE_BLANK":
            return str(candidate.get("sentence", "빈칸에 알맞은 말을 골라 보세요."))
        if training_type == "IMAGE_SENTENCE_MATCH":
            return str(candidate.get("imagePrompt", "그림에 알맞은 문장을 골라 보세요."))
        audio_text = candidate.get("audioText") or candidate.get("targetAudioText", "")
        return f"‘{audio_text}’를 듣고 골라 보세요."
    if training_type in BUILD_TYPES:
        return f"‘{candidate.get('targetAudioText', '')}’ 글자를 만들어 보세요."
    if training_type == "SYLLABLE_BLEND":
        return "음절 카드를 순서대로 합쳐 낱말을 만들어 보세요."
    if training_type == "FINAL_CONSONANT_DELETE":
        return f"‘{candidate.get('source', '')}’에서 알맞은 글자를 빼 보세요."
    if training_type == "SYLLABLE_DELETE":
        return f"‘{candidate.get('source', '')}’에서 알맞은 음절을 빼 보세요."
    if training_type == "SYLLABLE_REPLACE":
        return f"‘{candidate.get('source', '')}’의 음절을 바꿔 보세요."
    if training_type == "SENTENCE_ASSEMBLY":
        return "카드를 자연스러운 문장 순서로 이어 보세요."
    if training_type in READING_TYPES:
        return "화면의 글을 소리 내어 읽어 보세요."
    return "문제를 확인하고 답해 보세요."


def candidate_texts(candidate: Mapping[str, Any]) -> list[str]:
    if isinstance(candidate.get("sentences"), list):
        return [
            str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
            for value in candidate["sentences"]
        ]
    if isinstance(candidate.get("words"), list):
        return [
            str(value.get("text", "")) if isinstance(value, Mapping) else str(value)
            for value in candidate["words"]
        ]
    if isinstance(candidate.get("phrases"), list):
        return [str(value) for value in candidate["phrases"]]
    text = expected_text("", candidate)
    return [text] if text else []


__all__ = [
    "BUILD_TYPES",
    "CHOICE_TYPES",
    "READING_TYPES",
    "SERVICE_TEMPLATE_IDS",
    "SERVICE_TRAINING_CATALOG",
    "TRACE_TYPES",
    "build_candidate_request",
    "build_target_features",
    "candidate_texts",
    "choice_label",
    "correct_choice",
    "expected_text",
    "prompt_text",
    "service_training_groups",
    "service_training_spec_by_id",
    "service_training_spec_by_type",
]
