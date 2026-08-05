from __future__ import annotations

from copy import deepcopy
from typing import Any

CURRICULUM_SAMPLE_PROFILES: dict[str, dict[str, Any]] = {
    "신규 학생 · 근거 없음": {
        "description": "아직 수행 기록이 없어 가장 기초적인 균형 훈련으로 시작합니다.",
        "featureProfiles": [],
        "recentTrainings": [],
    },
    "자모 읽기가 어려운 학생": {
        "description": "기본 모음과 자음 정확도가 낮아 글자·소리 단계만 허용합니다.",
        "featureProfiles": [
            {
                "featureCode": "GRAPHEME.VOWEL.BASIC.ㅏ",
                "category": "GRAPHEME",
                "accuracyRate": 0.42,
                "weaknessScore": 0.82,
                "confidence": 0.90,
                "evidenceCount": 12,
                "pronunciationErrorRate": 0.36,
                "avgFixationDurationMs": 820,
                "avgRegressionCount": 1.4,
                "skipRate": 0.08,
            },
            {
                "featureCode": "GRAPHEME.ONSET.BASIC.ㄱ",
                "category": "GRAPHEME",
                "accuracyRate": 0.50,
                "weaknessScore": 0.75,
                "confidence": 0.85,
                "evidenceCount": 10,
                "pronunciationErrorRate": 0.30,
                "avgFixationDurationMs": 760,
                "avgRegressionCount": 1.2,
                "skipRate": 0.05,
            },
        ],
        "recentTrainings": [],
    },
    "음절 결합이 어려운 학생": {
        "description": "자모는 읽지만 음절을 합치고 분리하는 수행이 불안정합니다.",
        "featureProfiles": [
            {
                "featureCode": "GRAPHEME.ONSET.BASIC.ㄱ",
                "category": "GRAPHEME",
                "accuracyRate": 0.91,
                "weaknessScore": 0.18,
                "confidence": 0.92,
                "evidenceCount": 18,
            },
            {
                "featureCode": "SYLLABLE.CV",
                "category": "SYLLABLE",
                "accuracyRate": 0.58,
                "weaknessScore": 0.70,
                "confidence": 0.86,
                "evidenceCount": 14,
                "pronunciationErrorRate": 0.28,
                "avgFixationDurationMs": 900,
                "avgRegressionCount": 1.8,
                "skipRate": 0.06,
            },
        ],
        "recentTrainings": [
            {"trainingTemplateId": 16, "accuracy": 0.55, "daysAgo": 1},
        ],
    },
    "낱말 해독이 어려운 학생": {
        "description": "음절 조작은 가능하지만 낱말 해독 정확도와 시선 효율이 낮습니다.",
        "featureProfiles": [
            {
                "featureCode": "SYLLABLE.CV",
                "category": "SYLLABLE",
                "accuracyRate": 0.88,
                "weaknessScore": 0.22,
                "confidence": 0.90,
                "evidenceCount": 20,
            },
            {
                "featureCode": "WORD.SYLLABLE_COUNT.2",
                "category": "WORD",
                "accuracyRate": 0.61,
                "weaknessScore": 0.68,
                "confidence": 0.84,
                "evidenceCount": 16,
                "pronunciationErrorRate": 0.25,
                "avgFixationDurationMs": 1180,
                "avgRegressionCount": 2.8,
                "skipRate": 0.12,
            },
        ],
        "recentTrainings": [
            {"trainingTemplateId": 22, "accuracy": 0.60, "daysAgo": 1},
            {"trainingTemplateId": 22, "accuracy": 0.57, "daysAgo": 3},
        ],
    },
    "문장·유창성 단계 학생": {
        "description": "기초 해독은 안정적이고 문장 이해와 유창성을 확장할 단계입니다.",
        "featureProfiles": [
            {
                "featureCode": "WORD.SYLLABLE_COUNT.2",
                "category": "WORD",
                "accuracyRate": 0.90,
                "weaknessScore": 0.20,
                "confidence": 0.93,
                "evidenceCount": 24,
            },
            {
                "featureCode": "SENTENCE.SIMPLE",
                "category": "SENTENCE",
                "accuracyRate": 0.72,
                "weaknessScore": 0.52,
                "confidence": 0.88,
                "evidenceCount": 18,
                "avgFixationDurationMs": 1050,
                "avgRegressionCount": 2.2,
                "skipRate": 0.09,
            },
        ],
        "recentTrainings": [
            {"trainingTemplateId": 27, "accuracy": 0.70, "daysAgo": 2},
        ],
    },
}


def curriculum_sample(name: str) -> dict[str, Any]:
    return deepcopy(CURRICULUM_SAMPLE_PROFILES[name])


__all__ = ["CURRICULUM_SAMPLE_PROFILES", "curriculum_sample"]
