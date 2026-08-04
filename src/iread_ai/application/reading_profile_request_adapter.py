from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iread_ai.contracts.reading_profile import (
    ReadingFeatureProfile,
    StudentReadingProfileSnapshot,
)
from iread_ai.contracts.teacher_report import (
    TeacherReportAnalyzeRequest,
    TeacherReportGazeTrend,
)
from iread_ai.curriculum_models import (
    CurriculumRecommendRequest,
    RecentCurriculumTraining,
)


def build_teacher_report_request(
    *,
    request_id: str,
    snapshot: StudentReadingProfileSnapshot,
    feature_labels: Mapping[str, str],
    gaze_trend: TeacherReportGazeTrend | Mapping[str, Any],
) -> TeacherReportAnalyzeRequest:
    missing_labels = [
        profile.feature_code
        for profile in snapshot.feature_profiles
        if not str(feature_labels.get(profile.feature_code, "")).strip()
    ]
    if missing_labels:
        raise ValueError(
            "featureLabels are required for teacher report profiles: "
            + ", ".join(sorted(missing_labels))
        )

    profiles: list[dict[str, Any]] = []
    for profile in snapshot.feature_profiles:
        payload = _shared_profile_payload(profile)
        payload["featureLabel"] = str(feature_labels[profile.feature_code]).strip()
        profiles.append(payload)

    return TeacherReportAnalyzeRequest.model_validate(
        {
            "requestId": request_id,
            "schemaVersion": 1,
            "profileAnalysisVersion": snapshot.profile_analysis_version,
            "featureProfiles": profiles,
            "gazeTrend": _json_payload(gaze_trend),
        }
    )


def build_curriculum_recommend_request(
    *,
    request_id: str,
    snapshot: StudentReadingProfileSnapshot,
    recent_trainings: Sequence[RecentCurriculumTraining | Mapping[str, Any]] = (),
    current_stage_hint: int | None = None,
    use_llm: bool = True,
) -> CurriculumRecommendRequest:
    payload: dict[str, Any] = {
        "requestId": request_id,
        "schemaVersion": 1,
        "featureProfiles": [
            _shared_profile_payload(profile) for profile in snapshot.feature_profiles
        ],
        "recentTrainings": [_json_payload(item) for item in recent_trainings],
        "useLlm": use_llm,
    }
    if current_stage_hint is not None:
        payload["currentStageHint"] = current_stage_hint
    return CurriculumRecommendRequest.model_validate(payload)


def _shared_profile_payload(profile: ReadingFeatureProfile) -> dict[str, Any]:
    return profile.model_dump(
        mode="json",
        by_alias=True,
        include=set(ReadingFeatureProfile.model_fields),
        exclude_none=True,
    )


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


__all__ = [
    "build_curriculum_recommend_request",
    "build_teacher_report_request",
]
