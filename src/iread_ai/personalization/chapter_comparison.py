from __future__ import annotations

import math
import time
import uuid
from typing import Protocol

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.contracts.story_page import StoryPageQualityPayload


class ChapterGenerationService(Protocol):
    async def generate(
        self,
        request: StoryChapterGenerateRequest,
    ) -> StoryChapterGenerateResponse: ...


class ChapterGenerationComparisonService:
    def __init__(
        self,
        *,
        baseline_service: ChapterGenerationService,
    ) -> None:
        self._baseline_service = baseline_service

    async def compare_displayed_chapter_to_plain(
        self,
        request: StoryChapterGenerateRequest,
        personalized_response: StoryChapterGenerateResponse,
    ) -> dict[str, object]:
        personalized_response.validate_against_request(request)
        started = time.perf_counter()
        baseline_response = await self._baseline_service.generate(request)
        baseline_response.validate_against_request(request)
        if baseline_response.generation.api_call_count != 1:
            raise RuntimeError(
                "baseline chapter comparison must make exactly one model call"
            )
        if baseline_response.generation.repair_attempted:
            raise RuntimeError(
                "baseline chapter comparison must not attempt repair"
            )

        plain = _outcome(
            baseline_response,
            request,
            pipeline="PLAIN",
        )
        personalized = _outcome(
            personalized_response,
            request,
            pipeline="PERSONALIZED",
        )
        comparison = _compare_outcomes(plain, personalized)
        wall_ms = (time.perf_counter() - started) * 1000
        return {
            "comparisonVersion": "displayed-chapter-vs-plain-v1",
            "plain": plain,
            "personalized": personalized,
            "comparison": comparison,
            "diagnostics": {
                "baselineApiCallCount": (
                    baseline_response.generation.api_call_count
                ),
                "personalizedApiCallCount": (
                    personalized_response.generation.api_call_count
                ),
                "newApiCallCount": 1,
                "baselineRepairAttempted": (
                    baseline_response.generation.repair_attempted
                ),
                "comparisonWallElapsedMs": round(wall_ms, 3),
                "execution": (
                    "one_new_plain_chapter_call_after_"
                    "personalized_chapter"
                ),
            },
        }


def _outcome(
    response: StoryChapterGenerateResponse,
    request: StoryChapterGenerateRequest,
    *,
    pipeline: str,
) -> dict[str, object]:
    quality = response.quality.chapter
    fit = _profile_fit(quality, request)
    return {
        "pipeline": pipeline,
        "status": "SUCCESS",
        "pages": [
            page.model_dump(mode="json", by_alias=True)
            for page in response.pages
        ],
        "fit": fit,
        "quality": response.quality.model_dump(
            mode="json",
            by_alias=True,
        ),
        "generation": response.generation.model_dump(
            mode="json",
            by_alias=True,
        ),
        "timingMs": response.timing_ms.model_dump(
            mode="json",
            by_alias=True,
        ),
    }


def _profile_fit(
    quality: StoryPageQualityPayload,
    request: StoryChapterGenerateRequest,
) -> dict[str, object]:
    quality_by_code = {
        skill.code: skill
        for skill in quality.per_skill
    }
    unverified_skill_codes = [
        policy.code
        for policy in request.generation_profile.skills
        if (
            policy.code.startswith("PHONO_")
            and quality.analysis_status != "FULL"
        )
        or (
            policy.code in quality_by_code
            and quality_by_code[policy.code].status == "UNVERIFIED"
        )
    ]
    total_skill_count = len(request.generation_profile.skills)
    verified_skill_count = max(
        0,
        total_skill_count - len(unverified_skill_codes),
    )
    score_confidence = (
        "PARTIAL" if unverified_skill_codes else "FULL"
    )
    surface_weighted_risk = sum(
        float(skill.weighted_risk)
        for skill in quality.per_skill
        if not skill.code.startswith("PHONO_")
    )
    surface_risk_per_10 = (
        10.0
        * surface_weighted_risk
        / max(1, quality.written_syllable_count)
    )
    full_score = 100.0 * math.exp(-0.55 * quality.risk_per_10)
    surface_score = 100.0 * math.exp(-0.55 * surface_risk_per_10)
    score = (
        surface_score
        if score_confidence == "PARTIAL"
        else full_score
    )
    if score_confidence == "PARTIAL":
        score_reason = (
            "G2P가 확정하지 못한 PHONO 규칙은 0회로 간주하지 않고 "
            "제외한 뒤, 확인 가능한 규칙만으로 참고 점수를 계산했습니다."
        )
    else:
        score_reason = (
            "동일한 아동 프로필로 Kiwi·G2P 분석 후 "
            "100×exp(-0.55×riskPer10)을 계산했습니다."
        )
    per_skill = [
        skill.model_dump(mode="json", by_alias=True)
        for skill in quality.per_skill
    ]
    return {
        "status": quality.status,
        "comparable": True,
        "analysisStatus": quality.analysis_status,
        "scoreConfidence": score_confidence,
        "scoreBasis": (
            "SURFACE_ONLY"
            if score_confidence == "PARTIAL"
            else "FULL_POLICY"
        ),
        "scoreCoveragePercent": (
            100.0
            if total_skill_count == 0
            else round(
                100.0 * verified_skill_count / total_skill_count,
                1,
            )
        ),
        "verifiedSkillCount": verified_skill_count,
        "totalSkillCount": total_skill_count,
        "unverifiedSkillCodes": unverified_skill_codes,
        "profileFitScore": round(score, 2),
        "surfaceProfileFitScore": round(surface_score, 2),
        "scoreReason": score_reason,
        "riskPer10": float(quality.risk_per_10),
        "surfaceRiskPer10": round(surface_risk_per_10, 6),
        "contractPass": quality.contract_pass,
        "contractFailures": list(quality.contract_failures),
        "excludedOverage": quality.excluded_overage_count,
        "limitedOverage": quality.limited_overage_count,
        "targetDistance": sum(
            int(skill.target_distance or 0)
            for skill in quality.per_skill
        ),
        "featureOccurrences": {
            skill.code: skill.occurrences
            for skill in quality.per_skill
        },
        "writtenSyllables": quality.written_syllable_count,
        "directDialogueCount": quality.direct_dialogue_count,
        "perSkill": per_skill,
    }


def _compare_outcomes(
    plain: dict[str, object],
    personalized: dict[str, object],
) -> dict[str, object]:
    plain_fit = _fit_document(plain)
    personalized_fit = _fit_document(personalized)
    plain_timing = _timing_document(plain)
    personalized_timing = _timing_document(personalized)
    plain_generation = _generation_document(plain)
    personalized_generation = _generation_document(personalized)
    budget_note = (
        f" 일반은 후보 {plain_generation['candidateCount']}개·"
        f"모델 호출 {plain_generation['apiCallCount']}회, 개인화는 "
        f"후보 {personalized_generation['candidateCount']}개·"
        f"모델 호출 {personalized_generation['apiCallCount']}회의 "
        "실제 표시 결과라 생성 예산 차이도 함께 포함됩니다."
    )
    comparison_confidence = (
        "FULL"
        if plain_fit.get("scoreConfidence") == "FULL"
        and personalized_fit.get("scoreConfidence") == "FULL"
        else "PARTIAL"
    )
    score_key = (
        "profileFitScore"
        if comparison_confidence == "FULL"
        else "surfaceProfileFitScore"
    )
    risk_key = (
        "riskPer10"
        if comparison_confidence == "FULL"
        else "surfaceRiskPer10"
    )
    plain_score = plain_fit[score_key]
    personalized_score = personalized_fit[score_key]
    comparable = isinstance(
        plain_score,
        int | float,
    ) and isinstance(personalized_score, int | float)
    if comparable:
        score_delta = float(personalized_score) - float(plain_score)
        if score_delta > 0.005:
            winner = "PERSONALIZED"
        elif score_delta < -0.005:
            winner = "PLAIN"
        else:
            winner = "TIE"
        if comparison_confidence == "FULL":
            comparison_reason = (
                "두 장을 동일한 아동 프로필과 Kiwi·G2P 기준으로 "
                "비교했습니다."
                + budget_note
            )
        else:
            comparison_reason = (
                "두 장 모두 점수를 표시하되, G2P가 확정하지 못한 "
                "PHONO 규칙은 제외하고 확인 가능한 규칙만 비교한 "
                "참고 결과입니다. 미검증 규칙을 0회로 판정하지 않습니다."
                + budget_note
            )
    else:
        score_delta = None
        winner = "UNVERIFIED"
        comparison_reason = (
            "PHONO 정책이 있지만 한쪽 이상의 G2P 분석이 FULL이 "
            "아니어서 점수를 직접 비교하지 않았습니다."
            + budget_note
        )
    return {
        "comparable": comparable,
        "comparisonConfidence": comparison_confidence,
        "scoreBasis": (
            "FULL_POLICY"
            if comparison_confidence == "FULL"
            else "COMMON_SURFACE_ONLY"
        ),
        "plainProfileFitScore": plain_score,
        "personalizedProfileFitScore": personalized_score,
        "winner": winner,
        "comparisonReason": comparison_reason,
        "delta": {
            "profileFitScore": (
                round(score_delta, 3)
                if score_delta is not None
                else None
            ),
            "riskPer10": round(
                float(personalized_fit[risk_key])
                - float(plain_fit[risk_key]),
                6,
            ),
            "excludedOverage": (
                int(personalized_fit["excludedOverage"])
                - int(plain_fit["excludedOverage"])
            ),
            "limitedOverage": (
                int(personalized_fit["limitedOverage"])
                - int(plain_fit["limitedOverage"])
            ),
            "targetDistance": (
                int(personalized_fit["targetDistance"])
                - int(plain_fit["targetDistance"])
            ),
            "totalElapsedMs": round(
                float(personalized_timing["total"])
                - float(plain_timing["total"]),
                3,
            ),
            "apiCallCount": (
                int(personalized_generation["apiCallCount"])
                - int(plain_generation["apiCallCount"])
            ),
        },
    }


def _fit_document(outcome: dict[str, object]) -> dict[str, object]:
    value = outcome.get("fit")
    if not isinstance(value, dict):
        raise TypeError("comparison outcome fit must be an object")
    return value


def _timing_document(outcome: dict[str, object]) -> dict[str, object]:
    value = outcome.get("timingMs")
    if not isinstance(value, dict):
        raise TypeError("comparison outcome timingMs must be an object")
    return value


def _generation_document(
    outcome: dict[str, object],
) -> dict[str, object]:
    value = outcome.get("generation")
    if not isinstance(value, dict):
        raise TypeError("comparison outcome generation must be an object")
    return value


def new_chapter_comparison_id() -> str:
    return f"displayed-chapter-comparison-{uuid.uuid4().hex}"


__all__ = [
    "ChapterGenerationComparisonService",
    "ChapterGenerationService",
    "new_chapter_comparison_id",
]
