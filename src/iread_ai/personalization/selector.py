from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from iread_ai.personalization.analyzer import (
    AnalysisStatus,
    CandidateAnalysis,
    KoreanReadingAnalyzer,
)

POLICY_ROLES = frozenset({"ALLOWED", "EXCLUDED", "LIMITED", "TARGET"})


@dataclass(frozen=True, slots=True)
class SkillPolicy:
    code: str
    role: str
    max_occurrences: int | None = None
    target_min: int | None = None
    target_max: int | None = None
    unit_penalty: float = 1.0

    def __post_init__(self) -> None:
        normalized_code = self.code.strip()
        normalized_role = self.role.strip().upper()
        if not normalized_code:
            raise ValueError("SkillPolicy.code must not be blank")
        if normalized_role not in POLICY_ROLES:
            raise ValueError(f"unsupported SkillPolicy.role: {self.role}")
        for name, value in (
            ("max_occurrences", self.max_occurrences),
            ("target_min", self.target_min),
            ("target_max", self.target_max),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be zero or greater")
        if (
            self.target_min is not None
            and self.target_max is not None
            and self.target_min > self.target_max
        ):
            raise ValueError("target_min must not exceed target_max")
        if self.unit_penalty < 0:
            raise ValueError("unit_penalty must be zero or greater")
        object.__setattr__(self, "code", normalized_code)
        object.__setattr__(self, "role", normalized_role)


@dataclass(frozen=True, slots=True)
class ContentContract:
    sentence_count: int = 4
    preferred_min_syllables: int = 55
    preferred_max_syllables: int = 70
    accepted_min_syllables: int = 50
    accepted_max_syllables: int = 75
    direct_dialogue: int = 1

    def __post_init__(self) -> None:
        if self.sentence_count < 1:
            raise ValueError("sentence_count must be at least one")
        if self.direct_dialogue < 0:
            raise ValueError("direct_dialogue must be zero or greater")
        if not (
            0
            <= self.accepted_min_syllables
            <= self.preferred_min_syllables
            <= self.preferred_max_syllables
            <= self.accepted_max_syllables
        ):
            raise ValueError("content contract syllable ranges are inconsistent")


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    skills: tuple[SkillPolicy, ...] = ()
    content_contract: ContentContract = field(default_factory=ContentContract)
    protected_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        codes = [skill.code for skill in self.skills]
        if len(codes) != len(set(codes)):
            raise ValueError("GenerationProfile contains duplicate skill codes")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    sentences: tuple[str, ...]
    analysis: CandidateAnalysis
    contract_pass: bool
    contract_failures: tuple[str, ...]
    excluded_overage: int
    limited_overage: int
    target_distance: int
    feature_occurrences: dict[str, int]
    feature_risks: dict[str, float]
    total_risk: float
    risk_per_10: float
    preferred_length_distance: int
    contract_penalty: int = 0

    def to_dict(self) -> dict[str, object]:
        violations = list(self.contract_failures)
        if self.excluded_overage:
            violations.append(f"EXCLUDED_OVERAGE:{self.excluded_overage}")
        if self.limited_overage:
            violations.append(f"LIMITED_OVERAGE:{self.limited_overage}")
        if self.target_distance:
            violations.append(f"TARGET_DISTANCE:{self.target_distance}")
        if not self.contract_pass:
            quality_status = "REJECTED"
        elif self.excluded_overage or self.limited_overage:
            quality_status = "BEST_EFFORT"
        elif self.analysis.status is not AnalysisStatus.FULL:
            quality_status = "REVIEW"
        else:
            quality_status = "PASS"
        reading_fit_score = 100.0 * math.exp(-0.55 * self.risk_per_10)
        return {
            "candidate_id": self.candidate_id,
            "sentences": list(self.sentences),
            "analysis": self.analysis.to_dict(),
            "contract_pass": self.contract_pass,
            "contract_failures": list(self.contract_failures),
            "excluded_overage": self.excluded_overage,
            "limited_overage": self.limited_overage,
            "target_distance": self.target_distance,
            "feature_occurrences": dict(self.feature_occurrences),
            "feature_risks": {
                code: round(risk, 6) for code, risk in self.feature_risks.items()
            },
            "total_risk": round(self.total_risk, 6),
            "risk_per_10": round(self.risk_per_10, 6),
            "preferred_length_distance": self.preferred_length_distance,
            "contractPenalty": self.contract_penalty,
            "qualityStatus": quality_status,
            "readingFitScore": round(reading_fit_score, 2),
            "violations": violations,
        }


def _range_distance(value: int, minimum: int, maximum: int) -> int:
    if value < minimum:
        return minimum - value
    if value > maximum:
        return value - maximum
    return 0


def _occurrences(analysis: CandidateAnalysis, code: str) -> int:
    if code.startswith("PHONO_"):
        return int(analysis.phonological_rule_counts.get(code, 0))
    return int(analysis.controllable_surface_feature_counts.get(code, 0))


def evaluate_candidate(
    candidate_id: str,
    sentences: tuple[str, ...],
    profile: GenerationProfile,
    analyzer: KoreanReadingAnalyzer,
) -> CandidateEvaluation:
    normalized_id = str(candidate_id).strip()
    if not normalized_id:
        raise ValueError("candidate_id must not be blank")
    sentence_list = tuple(str(sentence).strip() for sentence in sentences)
    analysis = analyzer.analyze(
        sentence_list,
        protected_terms=profile.protected_terms,
    )
    contract = profile.content_contract
    failures: list[str] = []
    if len(sentence_list) != contract.sentence_count:
        failures.append("SENTENCE_COUNT")
    if any(not sentence for sentence in sentence_list):
        failures.append("BLANK_SENTENCE")
    if not (
        contract.accepted_min_syllables
        <= analysis.written_syllables
        <= contract.accepted_max_syllables
    ):
        failures.append("WRITTEN_SYLLABLE_RANGE")
    if analysis.dialogue_sentence_count != contract.direct_dialogue:
        failures.append("DIRECT_DIALOGUE_COUNT")
    accepted_length_distance = _range_distance(
        analysis.written_syllables,
        contract.accepted_min_syllables,
        contract.accepted_max_syllables,
    )
    contract_penalty = (
        20 * abs(len(sentence_list) - contract.sentence_count)
        + 25 * sum(not sentence for sentence in sentence_list)
        + accepted_length_distance
        + 10 * abs(analysis.dialogue_sentence_count - contract.direct_dialogue)
    )

    excluded_overage = 0
    limited_overage = 0
    target_distance = 0
    occurrences: dict[str, int] = {}
    risks: dict[str, float] = {}
    for skill in profile.skills:
        count = _occurrences(analysis, skill.code)
        occurrences[skill.code] = count
        if skill.role == "EXCLUDED":
            excluded_overage += max(0, count - (skill.max_occurrences or 0))
            risks[skill.code] = count * skill.unit_penalty
        elif skill.role == "LIMITED":
            limited_overage += max(0, count - (skill.max_occurrences or 0))
            risks[skill.code] = count * skill.unit_penalty
        elif skill.role == "TARGET":
            minimum = skill.target_min if skill.target_min is not None else count
            maximum = skill.target_max if skill.target_max is not None else count
            distance = _range_distance(count, minimum, maximum)
            target_distance += distance
            risks[skill.code] = distance * skill.unit_penalty
        else:
            risks[skill.code] = 0.0

    total_risk = sum(risks.values())
    risk_per_10 = 10.0 * total_risk / max(1, analysis.written_syllables)
    preferred_distance = _range_distance(
        analysis.written_syllables,
        contract.preferred_min_syllables,
        contract.preferred_max_syllables,
    )
    return CandidateEvaluation(
        candidate_id=normalized_id,
        sentences=sentence_list,
        analysis=analysis,
        contract_pass=not failures,
        contract_failures=tuple(failures),
        excluded_overage=excluded_overage,
        limited_overage=limited_overage,
        target_distance=target_distance,
        feature_occurrences=occurrences,
        feature_risks=risks,
        total_risk=total_risk,
        risk_per_10=risk_per_10,
        preferred_length_distance=preferred_distance,
        contract_penalty=contract_penalty,
    )


def _analysis_rank(status: AnalysisStatus) -> int:
    return {
        AnalysisStatus.FULL: 0,
        AnalysisStatus.UNRELIABLE: 1,
        AnalysisStatus.SURFACE_ONLY: 2,
    }[status]


def select_best(evaluations: Iterable[CandidateEvaluation]) -> CandidateEvaluation:
    rows = tuple(evaluations)
    if not rows:
        raise ValueError("at least one candidate evaluation is required")
    return min(
        rows,
        key=lambda row: (
            not row.contract_pass,
            row.contract_penalty,
            row.excluded_overage,
            row.limited_overage,
            row.target_distance,
            _analysis_rank(row.analysis.status),
            row.risk_per_10,
            row.preferred_length_distance,
            row.candidate_id,
        ),
    )


__all__ = [
    "CandidateEvaluation",
    "ContentContract",
    "GenerationProfile",
    "SkillPolicy",
    "evaluate_candidate",
    "select_best",
]
