from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TeacherReportSample:
    sample_id: str
    title: str
    description: str
    source: str
    payload: dict[str, Any]


_NO_GAZE = {
    "status": "NO_DATA",
    "comparisonAvailable": False,
    "points": [],
    "failedSessionCount": 0,
}


TEACHER_REPORT_SAMPLES: tuple[TeacherReportSample, ...] = (
    TeacherReportSample(
        sample_id="balanced-progress",
        title="꾸준히 성장하는 균형형 학습자",
        description=(
            "정확도 향상, 반복되는 어려움, 검사 시선의 되읽기 감소가 함께 나타나는 "
            "일반적인 교수자 보고서 시나리오입니다."
        ),
        source=(
            "Backend demo-data/teacher-personas.sql의 2번 페르소나와 "
            "student_feature_profiles·gaze_analysis_results 산식을 바탕으로 구성"
        ),
        payload={
            "requestId": "demo-balanced-progress",
            "schemaVersion": 1,
            "profileAnalysisVersion": "PROFILE_ANALYSIS_V1",
            "featureProfiles": [
                {
                    "featureCode": "SYLLABLE_READING",
                    "featureLabel": "음절 읽기",
                    "accuracyRate": 0.79,
                    "avgPronunciationScore": 792,
                    "pronunciationErrorRate": 0.15,
                    "avgFixationDurationMs": 930,
                    "avgFixationCount": 2.5,
                    "avgRegressionCount": 1.4,
                    "skipRate": 0.08,
                    "avgReadingTimeMs": 1780,
                    "weaknessScore": 380,
                    "confidence": 0.90,
                    "evidenceCount": 22,
                    "previousAccuracyRate": 0.65,
                    "previousWeaknessScore": 540,
                },
                {
                    "featureCode": "LONG_SENTENCE_BREATHING",
                    "featureLabel": "긴 문장 호흡",
                    "accuracyRate": 0.58,
                    "avgPronunciationScore": 640,
                    "pronunciationErrorRate": 0.23,
                    "avgFixationDurationMs": 1100,
                    "avgFixationCount": 2.8,
                    "avgRegressionCount": 1.9,
                    "skipRate": 0.12,
                    "avgReadingTimeMs": 2350,
                    "weaknessScore": 680,
                    "confidence": 0.86,
                    "evidenceCount": 18,
                    "previousAccuracyRate": 0.55,
                    "previousWeaknessScore": 710,
                },
            ],
            "gazeTrend": {
                "training": deepcopy(_NO_GAZE),
                "test": {
                    "status": "AVAILABLE",
                    "comparisonAvailable": True,
                    "points": [
                        {
                            "observedAt": "2026-04-07T10:14:00+09:00",
                            "totalVisitedDurationMs": 50120,
                            "totalVisitedCount": 69,
                            "reverseReadCount": 10,
                            "avgVisitedDurationMs": 785,
                        },
                        {
                            "observedAt": "2026-07-28T11:14:00+09:00",
                            "totalVisitedDurationMs": 47320,
                            "totalVisitedCount": 63,
                            "reverseReadCount": 7,
                            "avgVisitedDurationMs": 750,
                        },
                    ],
                    "failedSessionCount": 0,
                },
            },
        },
    ),
    TeacherReportSample(
        sample_id="effortful-success",
        title="정확하지만 많은 노력이 드는 학습자",
        description=(
            "정확도는 높지만 고정 시간·고정 횟수·되읽기·읽기 시간이 큰 경우를 "
            "정답률만으로 놓치지 않는지 확인합니다."
        ),
        source=(
            "Backend 6번 신중형 페르소나를 기반으로 V1 노력형 성공 경계값을 검증하도록 "
            "시선 부담 수치를 확대한 합성 시나리오"
        ),
        payload={
            "requestId": "demo-effortful-success",
            "schemaVersion": 1,
            "profileAnalysisVersion": "PROFILE_ANALYSIS_V1",
            "featureProfiles": [
                {
                    "featureCode": "READING_AUTOMATICITY",
                    "featureLabel": "읽기 자동화 속도",
                    "accuracyRate": 0.90,
                    "avgPronunciationScore": 910,
                    "pronunciationErrorRate": 0.04,
                    "avgFixationDurationMs": 1500,
                    "avgFixationCount": 4.0,
                    "avgRegressionCount": 2.4,
                    "skipRate": 0.02,
                    "avgReadingTimeMs": 2850,
                    "weaknessScore": 420,
                    "confidence": 0.88,
                    "evidenceCount": 16,
                    "previousAccuracyRate": 0.87,
                    "previousWeaknessScore": 450,
                }
            ],
            "gazeTrend": {
                "training": {
                    "status": "AVAILABLE",
                    "comparisonAvailable": False,
                    "points": [
                        {
                            "observedAt": "2026-07-08T15:13:00+09:00",
                            "totalVisitedDurationMs": 55400,
                            "totalVisitedCount": 81,
                            "reverseReadCount": 5,
                            "avgVisitedDurationMs": 1500,
                        }
                    ],
                    "failedSessionCount": 0,
                },
                "test": deepcopy(_NO_GAZE),
            },
        },
    ),
    TeacherReportSample(
        sample_id="insufficient-evidence",
        title="근거가 아직 부족한 신규 학습자",
        description=(
            "근거 수와 신뢰도가 최소 기준에 못 미치고 시선 데이터도 없을 때, "
            "성급한 향상·어려움 판단을 만들지 않는지 확인합니다."
        ),
        source=(
            "Backend 3번 신규 전입 페르소나를 기반으로 evidenceCount와 confidence를 "
            "최소 기준 미만으로 둔 안전성 시나리오"
        ),
        payload={
            "requestId": "demo-insufficient-evidence",
            "schemaVersion": 1,
            "profileAnalysisVersion": "PROFILE_ANALYSIS_V1",
            "featureProfiles": [
                {
                    "featureCode": "LETTER_RECOGNITION",
                    "featureLabel": "낱글자 인식",
                    "accuracyRate": 0.76,
                    "avgPronunciationScore": 750,
                    "pronunciationErrorRate": 0.10,
                    "avgFixationDurationMs": 760,
                    "avgFixationCount": 1.6,
                    "avgRegressionCount": 0.7,
                    "skipRate": 0.04,
                    "avgReadingTimeMs": 1220,
                    "weaknessScore": 350,
                    "confidence": 0.20,
                    "evidenceCount": 2,
                    "previousAccuracyRate": 0.60,
                    "previousWeaknessScore": 510,
                }
            ],
            "gazeTrend": {
                "training": deepcopy(_NO_GAZE),
                "test": deepcopy(_NO_GAZE),
            },
        },
    ),
    TeacherReportSample(
        sample_id="failed-gaze",
        title="시선 보정 실패가 포함된 학습자",
        description=(
            "프로필 분석은 유지하면서 실패한 시선 세션을 변화로 해석하지 않고 "
            "해석 보류 문장으로 반환하는지 확인합니다."
        ),
        source=(
            "Backend 5번 회복형 페르소나의 past_gaze_failure=true 사례를 기반으로 구성"
        ),
        payload={
            "requestId": "demo-failed-gaze",
            "schemaVersion": 1,
            "profileAnalysisVersion": "PROFILE_ANALYSIS_V1",
            "featureProfiles": [
                {
                    "featureCode": "GAZE_STABILITY",
                    "featureLabel": "시선 고정 안정화",
                    "accuracyRate": 0.69,
                    "avgPronunciationScore": 712,
                    "pronunciationErrorRate": 0.17,
                    "avgFixationDurationMs": 1040,
                    "avgFixationCount": 2.3,
                    "avgRegressionCount": 1.8,
                    "skipRate": 0.09,
                    "avgReadingTimeMs": 2100,
                    "weaknessScore": 510,
                    "confidence": 0.82,
                    "evidenceCount": 14,
                    "previousAccuracyRate": 0.55,
                    "previousWeaknessScore": 650,
                }
            ],
            "gazeTrend": {
                "training": {
                    "status": "FAILED",
                    "comparisonAvailable": False,
                    "points": [],
                    "failedSessionCount": 2,
                },
                "test": {
                    "status": "AVAILABLE",
                    "comparisonAvailable": False,
                    "points": [
                        {
                            "observedAt": "2026-07-28T14:14:00+09:00",
                            "totalVisitedDurationMs": 48600,
                            "totalVisitedCount": 67,
                            "reverseReadCount": 6,
                            "avgVisitedDurationMs": 860,
                        }
                    ],
                    "failedSessionCount": 0,
                },
            },
        },
    ),
)


def get_teacher_report_samples() -> tuple[TeacherReportSample, ...]:
    return tuple(
        TeacherReportSample(
            sample_id=sample.sample_id,
            title=sample.title,
            description=sample.description,
            source=sample.source,
            payload=deepcopy(sample.payload),
        )
        for sample in TEACHER_REPORT_SAMPLES
    )


__all__ = [
    "TEACHER_REPORT_SAMPLES",
    "TeacherReportSample",
    "get_teacher_report_samples",
]
