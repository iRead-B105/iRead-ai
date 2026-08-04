from __future__ import annotations

from copy import deepcopy
from typing import Any

BACKEND_PROFILE_SAMPLE: dict[str, Any] = {
    "featureProfiles": [
        {
            "featureCode": "SYLLABLE.COMPLEX_CODA",
            "accuracyRate": 0.42,
            "avgPronunciationScore": 54.0,
            "avgFixationDurationMs": 1350,
            "avgFixationCount": 3.1,
            "avgRegressionCount": 2.4,
            "skipRate": 0.18,
            "avgReadingTimeMs": 2800,
            "weaknessScore": 0.73,
            "confidence": 0.82,
            "evidenceCount": 15,
            "status": "WEAK",
            "analysisVersion": "WEAKNESS_V1",
            "analyzedAt": "2026-08-04T12:00:00",
        },
        {
            "featureCode": "PHONOLOGY.LIAISON",
            "accuracyRate": 0.64,
            "avgPronunciationScore": 68.0,
            "avgFixationDurationMs": 980,
            "avgFixationCount": 2.2,
            "avgRegressionCount": 1.3,
            "skipRate": 0.07,
            "avgReadingTimeMs": 2100,
            "weaknessScore": 0.51,
            "confidence": 0.76,
            "evidenceCount": 11,
            "status": "WATCH",
            "analysisVersion": "WEAKNESS_V1",
            "analyzedAt": "2026-08-04T12:00:00",
        },
    ],
    "featureLabels": {
        "SYLLABLE.COMPLEX_CODA": "겹받침 음절 읽기",
        "PHONOLOGY.LIAISON": "연음 읽기",
    },
    "gazeTrend": {
        "training": {
            "status": "NO_DATA",
            "comparisonAvailable": False,
            "points": [],
            "failedSessionCount": 0,
        },
        "test": {
            "status": "NO_DATA",
            "comparisonAvailable": False,
            "points": [],
            "failedSessionCount": 0,
        },
    },
    "recentTrainings": [
        {"trainingTemplateId": 22, "accuracy": 0.58, "daysAgo": 1},
    ],
}


def backend_profile_sample() -> dict[str, Any]:
    return deepcopy(BACKEND_PROFILE_SAMPLE)


__all__ = ["BACKEND_PROFILE_SAMPLE", "backend_profile_sample"]
