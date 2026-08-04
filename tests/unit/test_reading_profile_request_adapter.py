from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from iread_ai.application.reading_profile_request_adapter import (
    build_curriculum_recommend_request,
    build_teacher_report_request,
)
from iread_ai.contracts.reading_profile import StudentReadingProfileSnapshot
from iread_ai.devtools.backend_profile_samples import backend_profile_sample


def _snapshot() -> tuple[dict, StudentReadingProfileSnapshot]:
    sample = backend_profile_sample()
    snapshot = StudentReadingProfileSnapshot.model_validate(
        {"featureProfiles": sample["featureProfiles"]}
    )
    return sample, snapshot


def test_backend_student_feature_profile_view_matches_snapshot_contract() -> None:
    _, snapshot = _snapshot()

    first = snapshot.feature_profiles[0]
    assert first.feature_code == "SYLLABLE.COMPLEX_CODA"
    assert first.avg_pronunciation_score == 54.0
    assert first.weakness_score == 0.73
    assert first.status == "WEAK"
    assert snapshot.profile_analysis_version == "WEAKNESS_V1"


def test_one_snapshot_builds_teacher_report_request_without_backend_only_fields() -> None:
    sample, snapshot = _snapshot()

    request = build_teacher_report_request(
        request_id="backend-profile-teacher-report",
        snapshot=snapshot,
        feature_labels=sample["featureLabels"],
        gaze_trend=sample["gazeTrend"],
    )
    payload = request.model_dump(mode="json", by_alias=True)

    assert request.profile_analysis_version == "WEAKNESS_V1"
    assert request.feature_profiles[0].feature_label == "겹받침 음절 읽기"
    assert request.feature_profiles[0].weakness_score == 0.73
    assert "status" not in payload["featureProfiles"][0]
    assert "analysisVersion" not in payload["featureProfiles"][0]
    assert "analyzedAt" not in payload["featureProfiles"][0]


def test_one_snapshot_builds_curriculum_request_with_recent_history() -> None:
    sample, snapshot = _snapshot()

    request = build_curriculum_recommend_request(
        request_id="backend-profile-curriculum",
        snapshot=snapshot,
        recent_trainings=sample["recentTrainings"],
        use_llm=False,
    )

    assert request.featureProfiles[0].feature_code == "SYLLABLE.COMPLEX_CODA"
    assert request.featureProfiles[0].weakness_score == 0.73
    assert request.recentTrainings[0].trainingTemplateId == 22
    assert request.useLlm is False


def test_teacher_report_adapter_requires_human_readable_feature_labels() -> None:
    sample, snapshot = _snapshot()
    labels = deepcopy(sample["featureLabels"])
    labels.pop("PHONOLOGY.LIAISON")

    with pytest.raises(ValueError, match="PHONOLOGY.LIAISON"):
        build_teacher_report_request(
            request_id="missing-feature-label",
            snapshot=snapshot,
            feature_labels=labels,
            gaze_trend=sample["gazeTrend"],
        )


def test_snapshot_rejects_mixed_backend_analysis_versions() -> None:
    sample = backend_profile_sample()
    sample["featureProfiles"][1]["analysisVersion"] = "WEAKNESS_V2"

    with pytest.raises(ValidationError, match="analysisVersion values must match"):
        StudentReadingProfileSnapshot.model_validate(
            {"featureProfiles": sample["featureProfiles"]}
        )
