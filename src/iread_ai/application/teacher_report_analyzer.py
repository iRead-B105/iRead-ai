from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from iread_ai.contracts.teacher_report import (
    DataSufficiency,
    TeacherReportAnalyzeRequest,
    TeacherReportFeatureProfile,
    TeacherReportGazeSeries,
)

FactCategory = Literal["improved", "persistent", "training_gaze", "test_gaze"]
FactDirection = Literal[
    "improved",
    "persistent",
    "effort",
    "increase",
    "decrease",
    "stable",
    "unavailable",
]

MIN_EVIDENCE_COUNT = 3
MIN_CONFIDENCE = 0.30
IMPROVEMENT_ACCURACY_DELTA = 0.10
IMPROVEMENT_WEAKNESS_DELTA = 0.10
PERSISTENT_ACCURACY_MAX = 0.60
PERSISTENT_WEAKNESS_MIN = 0.60
FIXATION_DURATION_THRESHOLD_MS = 1_200
FIXATION_COUNT_THRESHOLD = 3.0
REGRESSION_COUNT_THRESHOLD = 2.0
READING_TIME_THRESHOLD_MS = 2_500
MEANINGFUL_GAZE_CHANGE_RATE = 0.15
UNSAFE_SUBJECT_FRAGMENTS = (
    "난독증",
    "진단",
    "중증",
    "장애",
    "원인",
    "치료",
    "완치",
    "정상",
    "비정상",
    "확실",
    "반드시",
    "이전 지시",
    "지시를 무시",
    "프롬프트",
    "system",
    "assistant",
    "diagnos",
    "dyslexia",
)


@dataclass(frozen=True, slots=True)
class TeacherReportFact:
    evidence_id: str
    category: FactCategory
    subject: str
    text: str
    direction: FactDirection
    priority: float


@dataclass(frozen=True, slots=True)
class TeacherReportFacts:
    improved: tuple[TeacherReportFact, ...]
    persistent: tuple[TeacherReportFact, ...]
    training_gaze: tuple[TeacherReportFact, ...]
    test_gaze: tuple[TeacherReportFact, ...]
    data_sufficiency: DataSufficiency

    @property
    def all(self) -> tuple[TeacherReportFact, ...]:
        return self.improved + self.persistent + self.training_gaze + self.test_gaze


class TeacherReportAnalyzer:
    analysis_version = "TEACHER_REPORT_ANALYSIS_V1"

    def analyze(self, request: TeacherReportAnalyzeRequest) -> TeacherReportFacts:
        eligible = [profile for profile in request.feature_profiles if _is_eligible(profile)]
        improved = sorted(
            filter(None, (self._improvement_fact(profile) for profile in eligible)),
            key=lambda fact: (-fact.priority, fact.evidence_id),
        )[:5]

        persistent_candidates: list[TeacherReportFact] = []
        for profile in eligible:
            persistent = self._persistent_fact(profile)
            if persistent is not None:
                persistent_candidates.append(persistent)
            effort = self._effort_fact(profile)
            if effort is not None:
                persistent_candidates.append(effort)
        persistent = sorted(
            persistent_candidates,
            key=lambda fact: (-fact.priority, fact.evidence_id),
        )[:5]

        training_gaze = self._gaze_facts(
            request.gaze_trend.training,
            category="training_gaze",
            label="훈련 시선",
        )
        test_gaze = self._gaze_facts(
            request.gaze_trend.test,
            category="test_gaze",
            label="검사 시선",
        )
        evidence_total = sum(profile.evidence_count for profile in eligible)
        gaze_has_observation = any(
            series.status == "AVAILABLE" and bool(series.points)
            for series in (request.gaze_trend.training, request.gaze_trend.test)
        )
        if len(eligible) >= 2 and evidence_total >= 10:
            sufficiency: DataSufficiency = "SUFFICIENT"
        elif eligible or gaze_has_observation:
            sufficiency = "PARTIAL"
        else:
            sufficiency = "INSUFFICIENT"

        return TeacherReportFacts(
            improved=tuple(improved),
            persistent=tuple(persistent),
            training_gaze=training_gaze,
            test_gaze=test_gaze,
            data_sufficiency=sufficiency,
        )

    def _improvement_fact(
        self,
        profile: TeacherReportFeatureProfile,
    ) -> TeacherReportFact | None:
        label = _safe_subject(profile)
        accuracy_delta = (
            profile.accuracy_rate - profile.previous_accuracy_rate
            if profile.previous_accuracy_rate is not None
            else None
        )
        weakness_delta = (
            profile.previous_weakness_score - profile.weakness_score
            if profile.previous_weakness_score is not None
            else None
        )
        if accuracy_delta is not None and accuracy_delta >= IMPROVEMENT_ACCURACY_DELTA:
            previous = _percent(profile.previous_accuracy_rate)
            current = _percent(profile.accuracy_rate)
            change = current - previous
            return TeacherReportFact(
                evidence_id=_fact_id("improved", profile.feature_code),
                category="improved",
                subject=label,
                text=(
                    f"{label} 정확도가 이전 {previous}%에서 현재 {current}%로 "
                    f"{change}%p 상승해 향상 흐름이 확인됩니다."
                ),
                direction="improved",
                priority=accuracy_delta * profile.confidence,
            )
        if weakness_delta is not None and weakness_delta >= IMPROVEMENT_WEAKNESS_DELTA:
            return TeacherReportFact(
                evidence_id=_fact_id("improved", profile.feature_code),
                category="improved",
                subject=label,
                text=(f"{label}의 종합 어려움 지표가 이전보다 낮아져 긍정적인 변화가 관찰됩니다."),
                direction="improved",
                priority=weakness_delta * profile.confidence,
            )
        return None

    def _persistent_fact(
        self,
        profile: TeacherReportFeatureProfile,
    ) -> TeacherReportFact | None:
        if (
            profile.accuracy_rate > PERSISTENT_ACCURACY_MAX
            and profile.weakness_score < PERSISTENT_WEAKNESS_MIN
        ):
            return None
        label = _safe_subject(profile)
        accuracy = _percent(profile.accuracy_rate)
        return TeacherReportFact(
            evidence_id=_fact_id("persistent", profile.feature_code),
            category="persistent",
            subject=label,
            text=(
                f"{label}은 정확도 {accuracy}%이며 누적 근거 {profile.evidence_count}건에서 "
                "어려움이 반복되어, 다음 회기에서도 지속 관찰이 필요합니다."
            ),
            direction="persistent",
            priority=profile.weakness_score * profile.confidence,
        )

    def _effort_fact(
        self,
        profile: TeacherReportFeatureProfile,
    ) -> TeacherReportFact | None:
        if profile.accuracy_rate < 0.80:
            return None
        burden: list[str] = []
        if (
            profile.avg_fixation_duration_ms is not None
            and profile.avg_fixation_duration_ms >= FIXATION_DURATION_THRESHOLD_MS
        ):
            burden.append(f"평균 고정 시간 {profile.avg_fixation_duration_ms}ms")
        if (
            profile.avg_fixation_count is not None
            and profile.avg_fixation_count >= FIXATION_COUNT_THRESHOLD
        ):
            burden.append(f"평균 고정 횟수 {_number(profile.avg_fixation_count)}회")
        if (
            profile.avg_regression_count is not None
            and profile.avg_regression_count >= REGRESSION_COUNT_THRESHOLD
        ):
            burden.append(f"평균 회귀 {_number(profile.avg_regression_count)}회")
        if (
            profile.avg_reading_time_ms is not None
            and profile.avg_reading_time_ms >= READING_TIME_THRESHOLD_MS
        ):
            burden.append(f"평균 읽기 시간 {profile.avg_reading_time_ms}ms")
        if not burden:
            return None
        label = _safe_subject(profile)
        details = ", ".join(burden)
        return TeacherReportFact(
            evidence_id=_fact_id("effort", profile.feature_code),
            category="persistent",
            subject=label,
            text=(
                f"{label}에서는 정확도 {_percent(profile.accuracy_rate)}%로 높지만, "
                f"{details} 등의 부담 지표가 함께 관찰됩니다. 정답 도달 과정을 "
                "다음 회기에서도 지속 관찰할 필요가 있습니다."
            ),
            direction="effort",
            priority=0.5 * profile.confidence,
        )

    def _gaze_facts(
        self,
        series: TeacherReportGazeSeries,
        *,
        category: Literal["training_gaze", "test_gaze"],
        label: str,
    ) -> tuple[TeacherReportFact, ...]:
        if series.status == "NO_DATA":
            return (
                TeacherReportFact(
                    evidence_id=_fact_id(category, "no-data"),
                    category=category,
                    subject=label,
                    text=f"{label} 데이터가 없어 변화 해석을 보류합니다.",
                    direction="unavailable",
                    priority=1,
                ),
            )
        if series.status == "FAILED":
            return (
                TeacherReportFact(
                    evidence_id=_fact_id(category, "failed"),
                    category=category,
                    subject=label,
                    text=(
                        f"{label} 수집에 실패한 세션 {series.failed_session_count}건이 있어 "
                        "변화 해석을 보류합니다."
                    ),
                    direction="unavailable",
                    priority=1,
                ),
            )
        if len(series.points) == 1:
            return (
                TeacherReportFact(
                    evidence_id=_fact_id(category, "single-point"),
                    category=category,
                    subject=label,
                    text=f"{label} 데이터가 1회만 있어 변화 비교를 보류합니다.",
                    direction="unavailable",
                    priority=1,
                ),
            )

        first = series.points[0]
        latest = series.points[-1]
        facts: list[TeacherReportFact] = []
        duration_direction = _meaningful_direction(
            first.total_visited_duration_ms,
            latest.total_visited_duration_ms,
        )
        if duration_direction is not None:
            verb = "증가" if duration_direction == "increase" else "감소"
            facts.append(
                TeacherReportFact(
                    evidence_id=_fact_id(category, "duration"),
                    category=category,
                    subject=label,
                    text=(
                        f"{label}의 총 체류 시간이 {first.total_visited_duration_ms}ms에서 "
                        f"{latest.total_visited_duration_ms}ms로 {verb}했습니다. "
                        "동일한 난이도에서 변화가 유지되는지 추가 확인이 필요합니다."
                    ),
                    direction=duration_direction,
                    priority=_change_rate(
                        first.total_visited_duration_ms,
                        latest.total_visited_duration_ms,
                    ),
                )
            )
        if first.reverse_read_count != latest.reverse_read_count:
            direction: Literal["increase", "decrease"] = (
                "increase" if latest.reverse_read_count > first.reverse_read_count else "decrease"
            )
            verb = "증가" if direction == "increase" else "감소"
            facts.append(
                TeacherReportFact(
                    evidence_id=_fact_id(category, "regression"),
                    category=category,
                    subject=label,
                    text=(
                        f"{label}의 역행 읽기 횟수가 {first.reverse_read_count}회에서 "
                        f"{latest.reverse_read_count}회로 {verb}했습니다. "
                        "동일한 난이도에서 변화가 유지되는지 추가 확인이 필요합니다."
                    ),
                    direction=direction,
                    priority=abs(latest.reverse_read_count - first.reverse_read_count),
                )
            )
        if first.avg_visited_duration_ms is not None and latest.avg_visited_duration_ms is not None:
            average_direction = _meaningful_direction(
                first.avg_visited_duration_ms,
                latest.avg_visited_duration_ms,
            )
            if average_direction is not None:
                verb = "증가" if average_direction == "increase" else "감소"
                facts.append(
                    TeacherReportFact(
                        evidence_id=_fact_id(category, "average-duration"),
                        category=category,
                        subject=label,
                        text=(
                            f"{label}의 평균 체류 시간이 {first.avg_visited_duration_ms}ms에서 "
                            f"{latest.avg_visited_duration_ms}ms로 {verb}했습니다. "
                            "동일한 난이도에서 변화가 유지되는지 추가 확인이 필요합니다."
                        ),
                        direction=average_direction,
                        priority=_change_rate(
                            first.avg_visited_duration_ms,
                            latest.avg_visited_duration_ms,
                        ),
                    )
                )
        if not facts:
            facts.append(
                TeacherReportFact(
                    evidence_id=_fact_id(category, "stable"),
                    category=category,
                    subject=label,
                    text=(
                        f"{label}의 주요 지표에서 뚜렷한 변화가 관찰되지 않았습니다. "
                        "동일한 난이도에서 추가 관찰이 필요합니다."
                    ),
                    direction="stable",
                    priority=0,
                )
            )
        return tuple(sorted(facts, key=lambda fact: (-fact.priority, fact.evidence_id))[:3])


def _is_eligible(profile: TeacherReportFeatureProfile) -> bool:
    return profile.evidence_count >= MIN_EVIDENCE_COUNT and profile.confidence >= MIN_CONFIDENCE


def _fact_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{namespace}-{digest}"


def _safe_subject(profile: TeacherReportFeatureProfile) -> str:
    label = " ".join(profile.feature_label.split())
    lowered = label.casefold()
    if any(fragment in lowered for fragment in UNSAFE_SUBJECT_FRAGMENTS):
        digest = hashlib.sha256(profile.feature_code.encode("utf-8")).hexdigest()[:6]
        return f"읽기 특성 {digest}"
    return label


def _percent(value: float) -> int:
    return round(value * 100)


def _number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _change_rate(first: int, latest: int) -> float:
    return abs(latest - first) / max(first, 1)


def _meaningful_direction(
    first: int,
    latest: int,
) -> Literal["increase", "decrease"] | None:
    if first == latest or _change_rate(first, latest) < MEANINGFUL_GAZE_CHANGE_RATE:
        return None
    return "increase" if latest > first else "decrease"


__all__ = [
    "TeacherReportAnalyzer",
    "TeacherReportFact",
    "TeacherReportFacts",
]
