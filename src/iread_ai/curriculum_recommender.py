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

PROMPT_VERSION = "curriculum-rerank-v2"
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

_CATEGORY_FALLBACK_STAGE = {
    "GRAPHEME": 1,
    "SYLLABLE": 3,
    "PHONOLOGY": 6,
    "WORD": 6,
    "SENTENCE": 7,
}

_AGGREGATE_FEATURE_CODES = {
    "GRAPHEME",
    "GRAPHEME.ONSET",
    "GRAPHEME.VOWEL",
    "GRAPHEME.CODA",
    "SYLLABLE",
    "PHONOLOGY",
    "WORD",
    "SENTENCE",
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
    actionable_profiles = _actionable_profiles(
        request.featureProfiles,
        maximum_allowed_stage,
    )
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
            actionable_profiles,
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
                    provider,
                    planning_candidates,
                    deterministic,
                    actionable_profiles,
                    current_stage,
                    maximum_allowed_stage,
                )
                recommendation_provider = getattr(provider, "provider_name", "gms")
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
            profiles=actionable_profiles,
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
            f"교수자가 설정한 {request.currentStageHint}단계를 기준으로 다음 훈련을 구성했습니다.",
        )

    sufficient = [
        profile for profile in request.featureProfiles if _has_sufficient_evidence(profile)
    ]
    if not sufficient:
        return 1, "신뢰할 수 있는 수행 근거가 부족하여 가장 기초적인 1단계로 시작합니다."

    specific = [profile for profile in sufficient if _is_specific_feature(profile)]
    stage_evidence = specific or sufficient
    stable = [
        profile
        for profile in stage_evidence
        if profile.accuracy_rate >= MASTERY_ACCURACY
        and profile.weakness_score <= WEAKNESS_THRESHOLD
    ]
    unsettled = [
        profile
        for profile in stage_evidence
        if profile not in stable
    ]

    if not stable:
        stage = min((_profile_stage(profile) for profile in unsettled), default=1)
        next_stage_count = sum(_profile_stage(profile) > stage for profile in unsettled)
        next_stage_note = (
            " 아직 익숙하지 않은 다음 단계 요소는 부담이 되지 않도록 "
            "가벼운 도전 활동으로만 포함했습니다."
            if next_stage_count
            else ""
        )
        return (
            stage,
            f"최근 학습 기록에서 {stage}단계 읽기 요소를 아직 어려워하고 있어, "
            f"같은 단계를 충분히 연습하도록 구성했습니다.{next_stage_note}",
        )

    stable_frontier = max(_profile_stage(profile) for profile in stable)
    observed_frontier = max(_profile_stage(profile) for profile in stage_evidence)
    unsettled_at_frontier = any(
        _profile_stage(profile) == stable_frontier for profile in unsettled
    )
    if observed_frontier > stable_frontier:
        stage = min(observed_frontier, stable_frontier + 1)
        progression = (
            f"최근 학습 기록에서 {stable_frontier}단계까지 안정적으로 수행했습니다. "
            f"아직 충분히 확인되지 않은 상위 활동은 한 번에 높이지 않고 "
            f"{stage}단계 범위에서 천천히 연습하도록 구성했습니다."
        )
    elif unsettled_at_frontier:
        stage = stable_frontier
        progression = (
            f"{stable_frontier}단계에서 잘하는 부분과 어려워하는 부분이 함께 보여, "
            "같은 단계를 유지하면서 부족한 부분을 보완하도록 구성했습니다."
        )
    elif stable_frontier == 8:
        stage = 8
        progression = (
            "가장 높은 8단계 활동도 안정적으로 수행해, 현재 수준을 유지하면서 "
            "읽기 유창성을 높이는 훈련으로 구성했습니다."
        )
    else:
        stage = min(8, stable_frontier + 1)
        progression = (
            f"최근 학습 기록에서 {stable_frontier}단계까지 안정적으로 수행해, "
            f"다음 단계인 {stage}단계 활동을 중심으로 구성했습니다."
        )

    reinforcement_count = sum(
        _profile_stage(profile) < stage for profile in unsettled
    )
    reinforcement_note = (
        " 이전 단계에서 아직 어려워하는 부분은 복습 활동에 함께 넣었습니다."
        if reinforcement_count
        else ""
    )
    return stage, f"{progression}{reinforcement_note}"


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
    return profile.confidence >= MIN_CONFIDENCE and profile.evidence_count >= MIN_EVIDENCE_COUNT


def _actionable_profiles(
    profiles: list[CurriculumFeatureProfile],
    maximum_allowed_stage: int,
) -> list[CurriculumFeatureProfile]:
    return [
        profile
        for profile in profiles
        if _has_sufficient_evidence(profile)
        and _profile_stage(profile) <= maximum_allowed_stage
    ]


def _profile_stage(profile: CurriculumFeatureProfile) -> int:
    code = profile.feature_code.upper()
    if code.startswith(("GRAPHEME.ONSET.BASIC", "GRAPHEME.VOWEL.BASIC")):
        return 1
    if code.startswith(
        (
            "GRAPHEME.ONSET.TENSE",
            "GRAPHEME.ONSET.ASPIRATED",
            "GRAPHEME.VOWEL.COMPOUND",
            "GRAPHEME.CODA.SIMPLE",
        )
    ):
        return 2
    if code.startswith("GRAPHEME.CODA.COMPLEX"):
        return 3
    if code == "SYLLABLE.CV" or code.startswith("SYLLABLE.CV."):
        return 2
    if code.startswith("SYLLABLE."):
        return 3
    if code.startswith(("PHONOLOGY.", "WORD.")):
        return 6
    if code.startswith(("SENTENCE.PHRASE_BOUNDARY", "SENTENCE.FLUENCY")):
        return 8
    if code.startswith("SENTENCE."):
        return 7
    return _CATEGORY_FALLBACK_STAGE[_profile_category(profile)]


def _is_specific_feature(profile: CurriculumFeatureProfile) -> bool:
    return profile.feature_code.upper() not in _AGGREGATE_FEATURE_CODES


def _profile_category(profile: CurriculumFeatureProfile) -> str:
    if profile.category is not None:
        return profile.category
    prefix = profile.feature_code.split(".", 1)[0].upper()
    return prefix if prefix in _CATEGORY_FALLBACK_STAGE else "WORD"


def _score_candidate(
    spec: CurriculumTemplateSpec,
    profiles: list[CurriculumFeatureProfile],
    current_stage: int,
    recent_count: int,
) -> _ScoredCandidate:
    matched = _matched_profiles(spec, profiles)
    strongest = _profile_priority(matched[0]) if matched else 0.15
    second = _profile_priority(matched[1]) if len(matched) > 1 else 0.0
    stage_bonus = 0.14 if spec.stage == current_stage else 0.08
    if spec.stage < current_stage:
        stage_bonus = max(0.03, 0.10 - (current_stage - spec.stage) * 0.015)
    repeat_penalty = min(0.24, recent_count * 0.08)
    score = max(0.0, min(1.0, strongest + second * 0.12 + stage_bonus - repeat_penalty))

    reasons: list[RecommendationReasonCode] = []
    if matched and matched[0].weakness_score >= 0.6:
        reasons.append("HIGH_WEAKNESS")
    if matched and matched[0].accuracy_rate < 0.7:
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
        target_feature_codes=tuple(profile.feature_code for profile in matched[:3]),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _matched_profiles(
    spec: CurriculumTemplateSpec,
    profiles: list[CurriculumFeatureProfile],
) -> list[CurriculumFeatureProfile]:
    directly_related = [
        profile
        for profile in profiles
        if _feature_affinity(spec.suggested_feature, profile.feature_code) > 0
    ]
    if directly_related:
        return sorted(
            directly_related,
            key=lambda profile: (
                _feature_affinity(spec.suggested_feature, profile.feature_code),
                _profile_priority(profile),
            ),
            reverse=True,
        )[:3]

    category_fallback = [
        profile
        for profile in profiles
        if _profile_category(profile) in spec.supported_categories
    ]
    return sorted(category_fallback, key=_profile_priority, reverse=True)[:3]


def _feature_affinity(suggested_feature: str, profile_feature: str) -> int:
    if suggested_feature == profile_feature:
        return 4
    suggested_parts = suggested_feature.split(".")
    profile_parts = profile_feature.split(".")
    common_parts = 0
    for suggested_part, profile_part in zip(suggested_parts, profile_parts, strict=False):
        if suggested_part != profile_part:
            break
        common_parts += 1
    if common_parts >= 3:
        return 3
    if common_parts == 2:
        return 2
    return 0


def _profile_priority(profile: CurriculumFeatureProfile) -> float:
    evidence_weight = profile.confidence * min(1.0, profile.evidence_count / 10)
    return min(
        0.86,
        profile.weakness_score * 0.42
        + (1 - profile.accuracy_rate) * 0.24
        + evidence_weight * 0.10
        + (profile.pronunciation_error_rate or 0.0) * 0.08
        + _gaze_burden(profile) * 0.12,
    )


def _gaze_burden(profile: CurriculumFeatureProfile) -> float:
    fixation = 0.0
    if profile.avg_fixation_duration_ms is not None:
        fixation = max(0.0, min(1.0, (profile.avg_fixation_duration_ms - 400) / 1100))
    regression = min(1.0, (profile.avg_regression_count or 0.0) / 5)
    skip = profile.skip_rate or 0.0
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

    foundation_reinforcement = [
        item
        for item in ordered
        if item.spec.stage < current_stage and item.spec.template_id not in selected_ids
    ]
    take(foundation_reinforcement, 1, "REINFORCEMENT")

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
    provider: GMSTextProvider,
    scored: list[_ScoredCandidate],
    deterministic: list[
        tuple[_ScoredCandidate, RecommendationRole, str, tuple[RecommendationReasonCode, ...]]
    ],
    actionable_profiles: list[CurriculumFeatureProfile],
    current_stage: int,
    maximum_allowed_stage: int,
) -> list[tuple[_ScoredCandidate, RecommendationRole, str, tuple[RecommendationReasonCode, ...]]]:
    deterministic_ids = {item.spec.template_id for item, _, _, _ in deterministic}
    shortlist = sorted(scored, key=lambda item: item.score, reverse=True)[:12]
    shortlist_by_id = {item.spec.template_id: item for item in shortlist}
    for item, _, _, _ in deterministic:
        shortlist_by_id[item.spec.template_id] = item
    shortlist = list(shortlist_by_id.values())
    minimum_score_by_role = {
        role: min(
            item.score
            for item, baseline_role, _, _ in deterministic
            if baseline_role == role
        )
        for role in ("CORE", "REINFORCEMENT", "STRETCH")
    }
    allowed_roles_by_id = {
        item.spec.template_id: _allowed_roles(
            item,
            current_stage=current_stage,
            maximum_allowed_stage=maximum_allowed_stage,
            minimum_score_by_role=minimum_score_by_role,
        )
        for item in shortlist
    }
    for item, role, _, _ in deterministic:
        allowed_roles_by_id[item.spec.template_id] = tuple(
            dict.fromkeys((*allowed_roles_by_id[item.spec.template_id], role))
        )
    shortlist = [
        item for item in shortlist if allowed_roles_by_id[item.spec.template_id]
    ]
    shortlist_by_id = {item.spec.template_id: item for item in shortlist}

    document: dict[str, Any] = {
        "promptVersion": PROMPT_VERSION,
        "currentStage": current_stage,
        "maximumAllowedStage": maximum_allowed_stage,
        "requiredComposition": {"CORE": 3, "REINFORCEMENT": 1, "STRETCH": 1},
        "deterministicBaselineIds": sorted(deterministic_ids),
        "minimumScoreByRole": minimum_score_by_role,
        "studentEvidence": [
            {
                "featureCode": profile.feature_code,
                "category": _profile_category(profile),
                "profileStage": _profile_stage(profile),
                "accuracyRate": profile.accuracy_rate,
                "weaknessScore": profile.weakness_score,
                "confidence": profile.confidence,
                "evidenceCount": profile.evidence_count,
            }
            for profile in _select_llm_profiles(
                actionable_profiles,
                current_stage=current_stage,
            )
        ],
        "eligibleCandidates": [
            {
                "trainingTemplateId": item.spec.template_id,
                "trainingType": item.spec.training_type,
                "trainingName": item.spec.name,
                "curriculumStage": item.spec.stage,
                "score": item.score,
                "allowedRoles": list(allowed_roles_by_id[item.spec.template_id]),
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
            "각 훈련에는 allowedRoles에 표시된 역할만 부여할 수 있습니다. "
            "각 역할의 후보는 minimumScoreByRole에 표시된 기준 점수보다 낮으면 안 됩니다. "
            "maximumAllowedStage를 넘거나 선수 단계를 건너뛰면 안 됩니다. "
            "진단이나 치료 표현을 사용하지 말고, 입력 근거에 있는 사실만 간결하게 설명하세요. "
            "rationale은 교수자가 바로 이해할 수 있는 자연스러운 한국어 한 문장으로 쓰고, "
            "영문 코드·점수·내부 역할명(CORE 등)은 적지 마세요."
        ),
        user_prompt=json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    draft = LlmCurriculumDraft.model_validate(generated)
    _validate_llm_draft(
        draft,
        shortlist_by_id,
        current_stage,
        maximum_allowed_stage,
        minimum_score_by_role,
        allowed_roles_by_id,
    )
    return [
        (
            shortlist_by_id[selection.trainingTemplateId],
            selection.role,
            selection.rationale,
            tuple(selection.reasonCodes),
        )
        for selection in draft.selections
    ]


def _select_llm_profiles(
    profiles: list[CurriculumFeatureProfile],
    *,
    current_stage: int,
    limit: int = 12,
) -> list[CurriculumFeatureProfile]:
    return sorted(
        profiles,
        key=lambda profile: (
            _profile_stage(profile) == current_stage,
            profile.accuracy_rate < MASTERY_ACCURACY
            or profile.weakness_score >= WEAKNESS_THRESHOLD,
            -abs(_profile_stage(profile) - current_stage),
            _profile_priority(profile),
            profile.feature_code,
        ),
        reverse=True,
    )[:limit]


def _validate_llm_draft(
    draft: LlmCurriculumDraft,
    candidates: dict[int, _ScoredCandidate],
    current_stage: int,
    maximum_allowed_stage: int,
    minimum_score_by_role: dict[str, float],
    allowed_roles_by_id: dict[int, tuple[RecommendationRole, ...]],
) -> None:
    ids = [selection.trainingTemplateId for selection in draft.selections]
    if len(ids) != len(set(ids)) or any(template_id not in candidates for template_id in ids):
        raise ValueError("LLM selected duplicate or unavailable templates")
    roles = [selection.role for selection in draft.selections]
    if roles.count("CORE") != 3 or roles.count("REINFORCEMENT") != 1 or roles.count("STRETCH") != 1:
        raise ValueError("LLM did not preserve the 3+1+1 composition")
    for selection in draft.selections:
        candidate = candidates[selection.trainingTemplateId]
        if selection.role not in allowed_roles_by_id[selection.trainingTemplateId]:
            raise ValueError("LLM assigned a role that is not allowed for the candidate")
        if candidate.score < minimum_score_by_role[selection.role]:
            raise ValueError("LLM selected a candidate below the deterministic role baseline")
        if candidate.spec.stage > maximum_allowed_stage:
            raise ValueError("LLM crossed the prerequisite stage gate")
        if selection.role == "CORE" and not (
            max(1, current_stage - 1) <= candidate.spec.stage <= current_stage
        ):
            raise ValueError("LLM used a distant or unmastered stage as a core training")
        if any(word in selection.rationale.casefold() for word in PROHIBITED_RATIONALE_WORDS):
            raise ValueError("LLM rationale contained a diagnostic expression")


def _allowed_roles(
    candidate: _ScoredCandidate,
    *,
    current_stage: int,
    maximum_allowed_stage: int,
    minimum_score_by_role: dict[str, float],
) -> tuple[RecommendationRole, ...]:
    roles: list[RecommendationRole] = []
    minimum_core_stage = max(1, current_stage - 1)
    if (
        minimum_core_stage <= candidate.spec.stage <= current_stage
        and candidate.score >= minimum_score_by_role["CORE"]
    ):
        roles.append("CORE")
    if (
        candidate.spec.stage <= current_stage
        and candidate.score >= minimum_score_by_role["REINFORCEMENT"]
    ):
        roles.append("REINFORCEMENT")
    if (
        candidate.spec.stage == maximum_allowed_stage
        and candidate.score >= minimum_score_by_role["STRETCH"]
    ):
        roles.append("STRETCH")
    return tuple(roles)


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
        profile for profile in profiles if profile.feature_code in candidate.target_feature_codes
    ]
    accuracy = sum(profile.accuracy_rate for profile in targets) / len(targets) if targets else 0.5
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
        _feature_label(candidate.target_feature_codes[0])
        if candidate.target_feature_codes
        else "기초 읽기"
    )
    if role == "CORE":
        return f"{target}을 집중적으로 연습해 현재 단계의 읽기 정확도를 높이도록 배치했습니다."
    if role == "REINFORCEMENT":
        return f"{target}의 기초를 다시 확인하고 안정적으로 읽을 수 있도록 배치했습니다."
    return f"부담이 크지 않은 범위에서 {target}을 한 단계 확장해 연습하도록 배치했습니다."


def _feature_label(feature_code: str) -> str:
    exact_labels = {
        "PHONOLOGY.LIAISON.CODA_TO_SILENT_ONSET": "받침 뒤에 모음이 이어질 때의 소리 연결",
        "PHONOLOGY.LIAISON": "연음",
        "PHONOLOGY.ASPIRATION": "거센소리 변화",
        "PHONOLOGY.NASALIZATION": "비음화",
        "PHONOLOGY.PALATALIZATION": "구개음화",
        "PHONOLOGY.LIQUIDIZATION": "유음화",
        "PHONOLOGY.TENSIFICATION": "된소리되기",
        "PHONOLOGY.CODA_NEUTRALIZATION": "받침 대표음",
        "SYLLABLE.COMPLEX_CODA": "겹받침 음절",
        "WORD.DECODING": "낱말 읽기",
        "SENTENCE.FLUENCY": "문장 유창성",
    }
    if feature_code in exact_labels:
        return exact_labels[feature_code]
    final_part = feature_code.rsplit(".", 1)[-1]
    if feature_code.startswith("GRAPHEME.ONSET.TENSE."):
        return f"된소리 초성 {final_part}"
    if feature_code.startswith("GRAPHEME.ONSET.ASPIRATED."):
        return f"거센소리 초성 {final_part}"
    if feature_code.startswith("GRAPHEME.ONSET."):
        return f"첫소리 {final_part}"
    if feature_code.startswith("GRAPHEME.VOWEL."):
        return f"모음 {final_part}"
    if feature_code.startswith("GRAPHEME.CODA.COMPLEX."):
        return f"겹받침 {final_part}"
    if feature_code.startswith("GRAPHEME.CODA."):
        return f"받침 {final_part}"
    if feature_code.startswith("WORD.SYLLABLE_COUNT."):
        return f"{final_part}음절 낱말"
    return "읽기 요소"


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
