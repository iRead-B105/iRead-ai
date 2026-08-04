from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from iread_ai.contracts.teacher_report import TeacherReportAnalyzeRequest


def teacher_report_request_payload() -> dict[str, Any]:
    return {
        "requestId": "teacher-report-2026-08-03-1",
        "schemaVersion": 1,
        "profileAnalysisVersion": "WEAKNESS_V1",
        "featureProfiles": [
            {
                "featureCode": "GRAPHEME.ONSET.BASIC.ㄱ",
                "featureLabel": "초성 ㄱ 읽기",
                "accuracyRate": 0.82,
                "avgPronunciationScore": 84.0,
                "pronunciationErrorRate": 0.12,
                "avgFixationDurationMs": 820,
                "avgFixationCount": 2.0,
                "avgRegressionCount": 0.8,
                "skipRate": 0.05,
                "avgReadingTimeMs": 1800,
                "weaknessScore": 0.25,
                "confidence": 0.85,
                "evidenceCount": 12,
                "previousAccuracyRate": 0.60,
                "previousWeaknessScore": 0.50,
            },
            {
                "featureCode": "GRAPHEME.CODA.SIMPLE.ㄴ",
                "featureLabel": "받침 ㄴ 읽기",
                "accuracyRate": 0.45,
                "avgPronunciationScore": 61.0,
                "pronunciationErrorRate": 0.38,
                "avgFixationDurationMs": 1450,
                "avgFixationCount": 4.0,
                "avgRegressionCount": 2.5,
                "skipRate": 0.20,
                "avgReadingTimeMs": 3100,
                "weaknessScore": 0.72,
                "confidence": 0.90,
                "evidenceCount": 15,
                "previousAccuracyRate": 0.48,
                "previousWeaknessScore": 0.70,
            },
        ],
        "gazeTrend": {
            "training": {
                "status": "AVAILABLE",
                "comparisonAvailable": True,
                "points": [
                    {
                        "observedAt": "2026-07-01T10:00:00+09:00",
                        "totalVisitedDurationMs": 5000,
                        "totalVisitedCount": 20,
                        "reverseReadCount": 4,
                        "avgVisitedDurationMs": 700,
                    },
                    {
                        "observedAt": "2026-08-01T10:00:00+09:00",
                        "totalVisitedDurationMs": 7000,
                        "totalVisitedCount": 23,
                        "reverseReadCount": 2,
                        "avgVisitedDurationMs": 800,
                    },
                ],
                "failedSessionCount": 0,
            },
            "test": {
                "status": "NO_DATA",
                "comparisonAvailable": False,
                "points": [],
                "failedSessionCount": 0,
            },
        },
    }


def test_teacher_report_contract_accepts_backend_aggregate_shape() -> None:
    request = TeacherReportAnalyzeRequest.model_validate(teacher_report_request_payload())

    assert request.request_id == "teacher-report-2026-08-03-1"
    assert request.feature_profiles[0].accuracy_rate == 0.82
    assert request.gaze_trend.training.comparison_available is True


def test_teacher_report_contract_rejects_duplicate_feature_codes() -> None:
    payload = teacher_report_request_payload()
    payload["featureProfiles"].append(deepcopy(payload["featureProfiles"][0]))

    with pytest.raises(ValidationError, match="featureCode values must be unique"):
        TeacherReportAnalyzeRequest.model_validate(payload)


def test_teacher_report_contract_uses_normalized_backend_score_ranges() -> None:
    request = TeacherReportAnalyzeRequest.model_validate(teacher_report_request_payload())

    assert request.feature_profiles[0].avg_pronunciation_score == 84.0
    assert request.feature_profiles[1].weakness_score == 0.72


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weaknessScore", 720),
        ("avgPronunciationScore", 610),
    ],
)
def test_teacher_report_contract_rejects_legacy_thousand_scale_scores(
    field: str,
    value: int,
) -> None:
    payload = teacher_report_request_payload()
    payload["featureProfiles"][0][field] = value

    with pytest.raises(ValidationError):
        TeacherReportAnalyzeRequest.model_validate(payload)


def test_teacher_report_contract_rejects_non_backend_feature_code() -> None:
    payload = teacher_report_request_payload()
    payload["featureProfiles"][0]["featureCode"] = "HAS_COMPOUND_VOWEL"

    with pytest.raises(ValidationError, match="Backend reading feature namespace"):
        TeacherReportAnalyzeRequest.model_validate(payload)


def test_teacher_report_contract_rejects_inconsistent_gaze_comparison_flag() -> None:
    payload = teacher_report_request_payload()
    payload["gazeTrend"]["training"]["comparisonAvailable"] = False

    with pytest.raises(ValidationError, match="comparisonAvailable"):
        TeacherReportAnalyzeRequest.model_validate(payload)


def test_teacher_report_contract_excludes_direct_student_identifiers() -> None:
    payload = teacher_report_request_payload()
    payload["studentId"] = 42

    with pytest.raises(ValidationError, match="Extra inputs"):
        TeacherReportAnalyzeRequest.model_validate(payload)
