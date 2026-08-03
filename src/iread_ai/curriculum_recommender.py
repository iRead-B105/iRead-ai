from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from iread_ai.curriculum_catalog import CURRICULUM_TEMPLATE_CATALOG, CurriculumTemplateSpec
from iread_ai.curriculum_models import (
    CurriculumCandidateAudit,
    CurriculumFeatureProfile,
    CurriculumRecommendation,
    CurriculumRecommendRequest,
    CurriculumRecommendResponse,
    DataSufficiency,
    LlmCurriculumDraft,
    RecommendationReasonCode,
    RecommendationRole,
)
from iread_ai.providers import GenerationProviderError, GMSTextProvider

logger = logging.getLogger(__name__)

PROMPT_VERSION = "curriculum-rerank-v1"
MIN_CONFIDENCE = 0.60
MIN_EVIDENCE_COUNT = 5
MASTERY_ACCURACY = 0.80
WEAKNESS_THRESHOLD = 0.35
REPEAT_COOLDOWN_DAYS = 3
REPEAT_SCORE_WINDOW_DAYS = 14
PROHIBITED_RATIONALE_WORDS = (
    "난독증",
    "진단",
    "장애",
    "치료",
    "중증",
    "dyslexia",
    "diagnosis",
    "disorder",
    "treatment",
)

_PROFILE_STAGE = {
    "GRAPHEME": 1,
    "SYLLABLE": 3,
    "PHONOLOGY": 6,
    "WORD": 6,
    "SENTENCE": 7,
}


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    spec: CurriculumTemplateSpec
    score: float
    target_feature_codes: tuple[str, ...]
    reason_codes: tuple[RecommendationReasonCode, ...]


def recommend_curriculum(
    request: CurriculumRecommendRequest,
    provider: GMSTextProvider | None,
) -> CurriculumRecommendResponse:
    data_sufficiency = _data_sufficiency(request.featureProfiles)
    current_stage, stage_rationale = _current_stage(request)
    maximum_allowed_stage = min(8, current_stage + 1)
    recent_counts = Counter(
        item.trainingTemplateId
        for item in request.recentTrainings
        if item.daysAgo <= REPEAT_SCORE_WINDOW_DAYS
    )
    cooldown_template_ids = {
        item.trainingTemplateId
        for item in request.recentTrainings
        if item.daysAgo <= REPEAT_COOLDOWN_DAYS
    }

    scored: list[_ScoredCandidate] = []
    audit: list[CurriculumCandidateAudit] = []
    for spec in CURRICULUM_TEMPLATE_CATALOG:
        if not spec.selectable:
            audit.append(_blocked_audit(spec, "RETIRED_TEMPLATE"))
            continue
        if spec.stage > maximum_allowed_stage:
            audit.append(_blocked_audit(spec, "PREREQUISITE_STAGE_NOT_REACHED"))
            continue
        candidate = _score_candidate(
            spec,
            request.featureProfiles,
            current_stage,
            recent_counts[spec.template_id],
        )
        scored.append(candidate)
        audit.append(
            CurriculumCandidateAudit(
                trainingTemplateId=spec.template_id,
                trainingType=spec.training_type,
                trainingName=spec.name,
                curriculumStage=spec.stage,
                status="ELIGIBLE",
                score=candidate.score,
                reasonCode=(
                    "RECENT_COOLDOWN_NOT_PREFERRED"
                    if spec.template_id in cooldown_template_ids
                    else "STAGE_AND_PREREQUISITES_PASSED"
                ),
            )
        )

    planning_candidates = _apply_recent_cooldown(
        scored,
        current_stage,
        maximum_allowed_stage,
        cooldown_template_ids,
    )
    deterministic = _deterministic_plan(
        planning_candidates,
        current_stage,
        maximum_allowed_stage,
    )
    selections = deterministic
    recommendation_provider: str = "deterministic"
    warnings: list[str] = []

    if request.useLlm:
        if provider is None:
            recommendation_provider = "deterministic-fallback"
            warnings.append("LLM provider가 설정되지 않아 규칙 기반 추천을 사용했습니다.")
        else:
            try:
                selections = _llm_plan(
                    request,
                    provider,
                    planning_candidates,
                    deterministic,
                    current_stage,
                    maximum_allowed_stage,
                )
                recommendation_provider = "gms"
            except (GenerationProviderError, ValidationError, ValueError) as exception:
                recommendation_provider = "deterministic-fallback"
                warnings.append("LLM 추천 검증에 실패하여 규칙 기반 추천을 사용했습니다.")
                logger.warning(
                    "Curriculum reranker fallback request_id=%s error=%s",
                    request.requestId,
                    type(exception).__name__,
                )

    recommendations = [
        _render_recommendation(
            sequence=index,
            candidate=candidate,
            role=role,
            profiles=request.featureProfiles,
            rationale=rationale,
            reason_codes=reason_codes,
        )
        for index, (candidate, role, rationale, reason_codes) in enumerate(
            selections,
            start=1,
        )
    ]
    if data_sufficiency != "SUFFICIENT":
        warnings.append("근거가 충분하지 않아 보수적인 기초 단계 추천을 적용했습니다.")

    return CurriculumRecommendResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        recommendationProvider=recommendation_provider,  # type: ignore[arg-type]
        dataSufficiency=data_sufficiency,
        currentStage=current_stage,
        maximumAllowedStage=maximum_allowed_stage,
        stageRationale=stage_rationale,
        recommendations=recommendations,
        candidateAudit=audit,
        warnings=list(dict.fromkeys(warnings)),
    )


def _current_stage(request: CurriculumRecommendRequest) -> tuple[int, str]:
    if request.currentStageHint is not None:
        return (
            request.currentStageHint,
            f"요청에서 제공한 현재 단계 {request.currentStageHint}을 적용했습니다.",
        )

    sufficient = [
        profile for profile in request.featureProfiles if _has_sufficient_evidence(profile)
    ]
    if not sufficient:
        return 1, "신뢰할 수 있는 수행 근거가 부족하여 가장 기초적인 1단계로 시작합니다."

    struggling = [
        profile
        for profile in sufficient
        if profile.accuracyRate < MASTERY_ACCURACY or profile.weaknessScore >= WEAKNESS_THRESHOLD
    ]
    if struggling:
        stage = min(_profile_stage(profile) for profile in struggling)
        feature = min(struggling, key=_profile_stage)
        return (
            stage,
            f"{feature.featureCode}의 기초 수행 근거를 우선해 현재 단계를 {stage}로 판정했습니다.",
        )

    stage = min(8, max(_profile_stage(profile) for profile in sufficient) + 1)
    return stage, f"관찰된 하위 특징이 안정적이어서 다음 학습 단계인 {stage}를 적용했습니다."


def _data_sufficiency(profiles: list[CurriculumFeatureProfile]) -> DataSufficiency:
    if not profiles:
        return "INSUFFICIENT"
    sufficient_count = sum(_has_sufficient_evidence(profile) for profile in profiles)
    if sufficient_count == len(profiles):
        return "SUFFICIENT"
    if sufficient_count:
        return "PARTIAL"
    return "INSUFFICIENT"


def _has_sufficient_evidence(profile: CurriculumFeatureProfile) -> bool:
    return profile.confidence >= MIN_CONFIDENCE and profile.evidenceCount >= MIN_EVIDENCE_COUNT


def _profile_stage(profile: CurriculumFeatureProfile) -> int:
    category = _profile_category(profile)
    return _PROFILE_STAGE[category]


def _profile_category(profile: CurriculumFeatureProfile) -> str:
    if profile.category is not None:
        return profile.category
    prefix = profile.featureCode.split(".", 1)[0].upper()
    return prefix if prefix in _PROFILE_STAGE else "WORD"


def _score_candidate(
    spec: CurriculumTemplateSpec,
    profiles: list[CurriculumFeatureProfile],
    current_stage: int,
    recent_count: int,
) -> _ScoredCandidate:
    matched = [
        profile for profile in profiles if _profile_category(profile) in spec.supported_categories
    ]
    matched.sort(key=_profile_priority, reverse=True)
    strongest = _profile_priority(matched[0]) if matched else 0.15
    second = _profile_priority(matched[1]) if len(matched) > 1 else 0.0
    stage_bonus = 0.14 if spec.stage == current_stage else 0.08
    if spec.stage < current_stage:
        stage_bonus = max(0.03, 0.10 - (current_stage - spec.stage) * 0.015)
    repeat_penalty = min(0.24, recent_count * 0.08)
    score = max(0.0, min(1.0, strongest + second * 0.12 + stage_bonus - repeat_penalty))

    reasons: list[RecommendationReasonCode] = []
    if matched and matched[0].weaknessScore >= 0.6:
        reasons.append("HIGH_WEAKNESS")
    if matched and matched[0].accuracyRate < 0.7:
        reasons.append("LOW_ACCURACY")
    if matched and _has_sufficient_evidence(matched[0]):
        reasons.append("RELIABLE_EVIDENCE")
    if matched and _gaze_burden(matched[0]) >= 0.5:
        reasons.append("GAZE_BURDEN")
    if spec.stage == current_stage:
        reasons.append("CURRENT_STAGE_MATCH")
    elif spec.stage < current_stage:
        reasons.append("FOUNDATION_REVIEW")
    else:
        reasons.append("NEXT_STAGE_SCAFFOLD")
    if recent_count:
        reasons.append("RECENT_REPEAT_PENALTY")
    if not reasons:
        reasons.append("LIMITED_EVIDENCE")

    return _ScoredCandidate(
        spec=spec,
        score=round(score, 4),
        target_feature_codes=tuple(profile.featureCode for profile in matched[:3]),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _profile_priority(profile: CurriculumFeatureProfile) -> float:
    evidence_weight = profile.confidence * min(1.0, profile.evidenceCount / 10)
    return min(
        0.86,
        profile.weaknessScore * 0.42
        + (1 - profile.accuracyRate) * 0.24
        + evidence_weight * 0.10
        + (profile.pronunciationErrorRate or 0.0) * 0.08
        + _gaze_burden(profile) * 0.12,
    )


def _gaze_burden(profile: CurriculumFeatureProfile) -> float:
    fixation = 0.0
    if profile.avgFixationDurationMs is not None:
        fixation = max(0.0, min(1.0, (profile.avgFixationDurationMs - 400) / 1100))
    regression = min(1.0, (profile.avgRegressionCount or 0.0) / 5)
    skip = profile.skipRate or 0.0
    return (fixation + regression + skip) / 3


def _deterministic_plan(
    scored: list[_ScoredCandidate],
    current_stage: int,
    maximum_allowed_stage: int,
) -> list[tuple[_ScoredCandidate, RecommendationRole, str, tuple[RecommendationReasonCode, ...]]]:
    ordered = sorted(
        scored,
        key=lambda item: (
            item.score,
            -abs(item.spec.stage - current_stage),
            -item.spec.template_id,
        ),
        reverse=True,
    )
    selected: list[tuple[_ScoredCandidate, RecommendationRole]] = []
    selected_ids: set[int] = set()

    def take(pool: list[_ScoredCandidate], count: int, role: RecommendationRole) -> None:
        for item in pool:
            if (
                len([selected_role for _, selected_role in selected if selected_role == role])
                >= count
            ):
                return
            if item.spec.template_id not in selected_ids:
                selected.append((item, role))
                selected_ids.add(item.spec.template_id)

    minimum_core_stage = max(1, current_stage - 1)
    core = [item for item in ordered if minimum_core_stage <= item.spec.stage <= current_stage]
    take(core, 3, "CORE")
    take(ordered, 3, "CORE")

    reinforcement = [
        item
        for item in ordered
        if item.spec.stage <= current_stage and item.spec.template_id not in selected_ids
    ]
    take(reinforcement, 1, "REINFORCEMENT")
    take(ordered, 1, "REINFORCEMENT")

    stretch = [
        item
        for item in ordered
        if item.spec.stage == maximum_allowed_stage and item.spec.template_id not in selected_ids
    ]
    take(stretch, 1, "STRETCH")
    take(ordered, 1, "STRETCH")

    if len(selected) != 5:
        raise ValueError("eligible curriculum catalog could not produce five recommendations")

    return [
        (
            item,
            role,
            _deterministic_rationale(item, role),
            _role_reason_codes(item, role),
        )
        for item, role in selected
    ]


def _llm_plan(
    request: CurriculumRecommendRequest,
    provider: GMSTextProvider,
    scored: list[_ScoredCandidate],
    deterministic: list[
        tuple[_ScoredCandidate, RecommendationRole, str, tuple[RecommendationReasonCode, ...]]
    ],
    current_stage: int,
    maximum_allowed_stage: int,
) -> list[tuple[_ScoredCandidate, RecommendationRole, str, tuple[RecommendationReasonCode, ...]]]:
    deterministic_ids = {item.spec.template_id for item, _, _, _ in deterministic}
    shortlist = sorted(scored, key=lambda item: item.score, reverse=True)[:12]
    shortlist_by_id = {item.spec.template_id: item for item in shortlist}
    for item, _, _, _ in deterministic:
        shortlist_by_id[item.spec.template_id] = item
    shortlist = list(shortlist_by_id.values())

    document: dict[str, Any] = {
        "promptVersion": PROMPT_VERSION,
        "currentStage": current_stage,
        "maximumAllowedStage": maximum_allowed_stage,
        "requiredComposition": {"CORE": 3, "REINFORCEMENT": 1, "STRETCH": 1},
        "deterministicBaselineIds": sorted(deterministic_ids),
        "studentEvidence": [
            {
                "featureCode": profile.featureCode,
                "category": _profile_category(profile),
                "accuracyRate": profile.accuracyRate,
                "weaknessScore": profile.weaknessScore,
                "confidence": profile.confidence,
                "evidenceCount": profile.evidenceCount,
            }
            for profile in request.featureProfiles
        ],
        "eligibleCandidates": [
            {
                "trainingTemplateId": item.spec.template_id,
                "trainingType": item.spec.training_type,
                "trainingName": item.spec.name,
                "curriculumStage": item.spec.stage,
                "score": item.score,
                "targetFeatureCodes": list(item.target_feature_codes),
                "reasonCodes": list(item.reason_codes),
            }
            for item in shortlist
        ],
    }
    generated = provider.generate_json(
        schema_name="curriculum_recommendation_v1",
        schema=LlmCurriculumDraft.model_json_schema(),
        system_prompt=(
            "당신은 초등 읽기 훈련의 일일 커리큘럼 재정렬 도우미입니다. "
            "eligibleCandidates에 있는 훈련만 사용하고 정확히 CORE 3개, "
            "REINFORCEMENT 1개, STRETCH 1개를 서로 다른 ID로 선택하세요. "
            "maximumAllowedStage를 넘거나 선수 단계를 건너뛰면 안 됩니다. "
            "진단이나 치료 표현을 사용하지 말고, 입력 근거에 있는 사실만 간결하게 설명하세요."
        ),
        user_prompt=json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    draft = LlmCurriculumDraft.model_validate(generated)
    _validate_llm_draft(draft, shortlist_by_id, current_stage, maximum_allowed_stage)
    return [
        (
            shortlist_by_id[selection.trainingTemplateId],
            selection.role,
            selection.rationale,
            tuple(selection.reasonCodes),
        )
        for selection in draft.selections
    ]


def _validate_llm_draft(
    draft: LlmCurriculumDraft,
    candidates: dict[int, _ScoredCandidate],
    current_stage: int,
    maximum_allowed_stage: int,
) -> None:
    ids = [selection.trainingTemplateId for selection in draft.selections]
    if len(ids) != len(set(ids)) or any(template_id not in candidates for template_id in ids):
        raise ValueError("LLM selected duplicate or unavailable templates")
    roles = [selection.role for selection in draft.selections]
    if roles.count("CORE") != 3 or roles.count("REINFORCEMENT") != 1 or roles.count("STRETCH") != 1:
        raise ValueError("LLM did not preserve the 3+1+1 composition")
    for selection in draft.selections:
        candidate = candidates[selection.trainingTemplateId]
        if candidate.spec.stage > maximum_allowed_stage:
            raise ValueError("LLM crossed the prerequisite stage gate")
        if selection.role == "CORE" and not (
            max(1, current_stage - 1) <= candidate.spec.stage <= current_stage
        ):
            raise ValueError("LLM used a distant or unmastered stage as a core training")
        if any(word in selection.rationale.casefold() for word in PROHIBITED_RATIONALE_WORDS):
            raise ValueError("LLM rationale contained a diagnostic expression")


def _render_recommendation(
    *,
    sequence: int,
    candidate: _ScoredCandidate,
    role: RecommendationRole,
    profiles: list[CurriculumFeatureProfile],
    rationale: str,
    reason_codes: tuple[RecommendationReasonCode, ...],
) -> CurriculumRecommendation:
    return CurriculumRecommendation(
        sequenceNo=sequence,
        trainingTemplateId=candidate.spec.template_id,
        trainingType=candidate.spec.training_type,
        trainingName=candidate.spec.name,
        curriculumStage=candidate.spec.stage,
        role=role,
        recommendedDifficulty=_recommended_difficulty(candidate, role, profiles),
        score=candidate.score,
        targetFeatureCodes=list(candidate.target_feature_codes),
        reasonCodes=list(reason_codes),
        rationale=rationale,
    )


def _recommended_difficulty(
    candidate: _ScoredCandidate,
    role: RecommendationRole,
    profiles: list[CurriculumFeatureProfile],
) -> int:
    targets = [
        profile for profile in profiles if profile.featureCode in candidate.target_feature_codes
    ]
    accuracy = sum(profile.accuracyRate for profile in targets) / len(targets) if targets else 0.5
    if accuracy < 0.60:
        difficulty = 1
    elif accuracy < 0.72:
        difficulty = 2
    elif accuracy < 0.84:
        difficulty = 3
    elif accuracy < 0.92:
        difficulty = 4
    else:
        difficulty = 5
    if role == "REINFORCEMENT":
        difficulty = max(1, difficulty - 1)
    elif role == "STRETCH":
        difficulty = min(5, difficulty + 1)
    return difficulty


def _role_reason_codes(
    candidate: _ScoredCandidate,
    role: RecommendationRole,
) -> tuple[RecommendationReasonCode, ...]:
    reasons = list(candidate.reason_codes)
    role_reason: RecommendationReasonCode = {
        "CORE": "CURRENT_STAGE_MATCH",
        "REINFORCEMENT": "FOUNDATION_REVIEW",
        "STRETCH": "NEXT_STAGE_SCAFFOLD",
    }[role]
    reasons.append(role_reason)
    return tuple(dict.fromkeys(reasons))[:5]


def _deterministic_rationale(candidate: _ScoredCandidate, role: RecommendationRole) -> str:
    target = (
        candidate.target_feature_codes[0] if candidate.target_feature_codes else "기초 읽기 특징"
    )
    if role == "CORE":
        return f"현재 단계에서 {target} 수행을 직접 보완하도록 배치했습니다."
    if role == "REINFORCEMENT":
        return f"{target}과 연결된 기초 기능을 안정적으로 반복하도록 배치했습니다."
    return f"선수 단계를 넘지 않는 범위에서 {target}을 다음 단계로 확장하도록 배치했습니다."


def _blocked_audit(spec: CurriculumTemplateSpec, reason: str) -> CurriculumCandidateAudit:
    return CurriculumCandidateAudit(
        trainingTemplateId=spec.template_id,
        trainingType=spec.training_type,
        trainingName=spec.name,
        curriculumStage=spec.stage,
        status="BLOCKED",
        reasonCode=reason,
    )


def _apply_recent_cooldown(
    scored: list[_ScoredCandidate],
    current_stage: int,
    maximum_allowed_stage: int,
    cooldown_template_ids: set[int],
) -> list[_ScoredCandidate]:
    if not cooldown_template_ids:
        return scored
    fresh = [item for item in scored if item.spec.template_id not in cooldown_template_ids]
    minimum_core_stage = max(1, current_stage - 1)
    core_count = sum(minimum_core_stage <= item.spec.stage <= current_stage for item in fresh)
    reinforcement_count = sum(item.spec.stage <= current_stage for item in fresh)
    if current_stage < 8:
        has_stretch = any(item.spec.stage == maximum_allowed_stage for item in fresh)
    else:
        has_stretch = len(fresh) >= 5
    if core_count >= 3 and reinforcement_count >= 4 and has_stretch:
        return fresh
    return scored


__all__ = ["recommend_curriculum"]
