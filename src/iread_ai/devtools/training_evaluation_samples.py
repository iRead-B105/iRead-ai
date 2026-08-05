from __future__ import annotations

from copy import deepcopy
from typing import Any

TRAINING_EVALUATION_SAMPLES: dict[str, dict[str, Any]] = {
    "객관식 3개 중 2개 정답": {
        "description": "정답 2개와 오답 1개를 동일한 비중으로 평가합니다.",
        "result": {
            "questions": [
                {"questionNo": 1, "isCorrect": True},
                {"questionNo": 2, "isCorrect": False},
                {"questionNo": 3, "isCorrect": True},
            ]
        },
    },
    "객관식 문항별 부분 점수": {
        "description": "백엔드의 0~1000 문항 점수를 0~100으로 환산합니다.",
        "result": {
            "questions": [
                {"questionNo": 1, "totalScore": 1000},
                {"questionNo": 2, "totalScore": 700},
                {"questionNo": 3, "totalScore": 400},
            ]
        },
    },
    "따라 읽기 재시도": {
        "description": "같은 문항을 재시도하면 attemptNo가 가장 큰 결과만 평가합니다.",
        "result": {
            "pronunciationAnalyses": [
                {
                    "questionNo": 1,
                    "referenceText": "토끼가 숲길을 천천히 걸어요.",
                    "pronunciationAccuracyScore": 54.0,
                    "fluencyScore": 62.0,
                    "completenessScore": 80.0,
                    "attemptNo": 1,
                    "passed": False,
                    "questionCompleted": False,
                },
                {
                    "questionNo": 1,
                    "referenceText": "토끼가 숲길을 천천히 걸어요.",
                    "pronunciationAccuracyScore": 82.0,
                    "fluencyScore": 76.0,
                    "completenessScore": 95.0,
                    "attemptNo": 2,
                    "passed": True,
                    "questionCompleted": True,
                },
                {
                    "questionNo": 2,
                    "referenceText": "친구와 함께 사과를 나누어 먹어요.",
                    "pronunciationAccuracyScore": 68.0,
                    "fluencyScore": 64.0,
                    "completenessScore": 88.0,
                    "attemptNo": 1,
                    "passed": False,
                    "questionCompleted": True,
                },
            ]
        },
    },
    "선택형과 따라 읽기 혼합": {
        "description": "선택형 문항 점수와 따라 읽기 정확도를 문항 단위로 평균냅니다.",
        "result": {
            "questions": [
                {"questionNo": 1, "totalScore": 900},
                {"questionNo": 2, "totalScore": 600},
            ],
            "pronunciationAnalyses": [
                {
                    "questionNo": 3,
                    "referenceText": "강아지가 공을 찾아 달려가요.",
                    "pronunciationAccuracyScore": 75.0,
                    "fluencyScore": 70.0,
                    "completenessScore": 100.0,
                    "attemptNo": 1,
                    "passed": True,
                    "questionCompleted": True,
                }
            ],
        },
    },
    "평가 근거 없음": {
        "description": "채점 가능한 근거가 없으면 100점이 아니라 0점을 반환합니다.",
        "result": {},
    },
}

READ_ALOUD_SENTENCES = (
    "토끼가 숲길을 천천히 걸어요.",
    "강아지가 공을 찾아 신나게 달려가요.",
    "친구와 함께 사과를 나누어 먹어요.",
)


def training_evaluation_sample(name: str) -> dict[str, Any]:
    return deepcopy(TRAINING_EVALUATION_SAMPLES[name])


__all__ = [
    "READ_ALOUD_SENTENCES",
    "TRAINING_EVALUATION_SAMPLES",
    "training_evaluation_sample",
]
