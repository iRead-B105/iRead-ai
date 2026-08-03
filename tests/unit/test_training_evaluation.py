from __future__ import annotations

import pytest

from iread_ai.training_evaluation import (
    TrainingEvaluationError,
    evaluate_training_result,
)


def test_objective_accuracy_uses_correctness_when_no_score_exists() -> None:
    breakdown = evaluate_training_result(
        {
            "questions": [
                {"questionNo": 1, "isCorrect": True},
                {"questionNo": 2, "isCorrect": False},
                {"questionNo": 3, "correctionConfirmed": True},
            ]
        }
    )

    assert breakdown.accuracy == 66.67
    assert [item.source for item in breakdown.question_scores] == [
        "OBJECTIVE",
        "OBJECTIVE",
        "OBJECTIVE",
    ]


def test_backend_total_score_is_normalized_from_thousand_point_scale() -> None:
    breakdown = evaluate_training_result(
        {
            "questions": [
                {"questionNo": 1, "totalScore": 1000},
                {"questionNo": 2, "totalScore": 700},
                {"questionNo": 3, "totalScore": 400},
            ]
        }
    )

    assert breakdown.accuracy == 70.0


def test_latest_pronunciation_attempt_replaces_earlier_attempt() -> None:
    breakdown = evaluate_training_result(
        {
            "pronunciationAnalyses": [
                {
                    "questionNo": 1,
                    "attemptNo": 1,
                    "pronunciationAccuracyScore": 40,
                },
                {
                    "questionNo": 1,
                    "attemptNo": 2,
                    "pronunciationAccuracyScore": 82,
                },
                {
                    "questionNo": 2,
                    "attemptNo": 1,
                    "pronunciationAccuracyScore": 68,
                },
            ]
        }
    )

    assert breakdown.accuracy == 75.0
    assert [item.score for item in breakdown.question_scores] == [82.0, 68.0]


def test_pronunciation_score_takes_precedence_for_same_question() -> None:
    breakdown = evaluate_training_result(
        {
            "questions": [
                {"questionNo": 1, "totalScore": 1000},
                {"questionNo": 2, "totalScore": 600},
            ],
            "pronunciationAnalyses": [
                {
                    "questionNo": 1,
                    "attemptNo": 1,
                    "pronunciationAccuracyScore": 75,
                }
            ],
        }
    )

    assert breakdown.accuracy == 67.5
    assert [item.source for item in breakdown.question_scores] == [
        "PRONUNCIATION",
        "OBJECTIVE",
    ]


def test_final_word_attempt_scores_are_fallback_evidence() -> None:
    breakdown = evaluate_training_result(
        {
            "wordAttempts": [
                {"isFinal": False, "totalScore": 300},
                {"isFinal": True, "totalScore": 850},
                {"isFinal": True, "totalScore": 650},
            ]
        }
    )

    assert breakdown.accuracy == 75.0
    assert all(item.source == "WORD_ATTEMPT" for item in breakdown.question_scores)


def test_empty_result_returns_zero_instead_of_perfect_score() -> None:
    breakdown = evaluate_training_result({})

    assert breakdown.accuracy == 0.0
    assert breakdown.warnings == ("NO_SCORABLE_EVIDENCE",)


def test_out_of_range_pronunciation_score_is_rejected() -> None:
    with pytest.raises(TrainingEvaluationError, match="between 0 and 100"):
        evaluate_training_result(
            {
                "pronunciationAnalyses": [
                    {
                        "questionNo": 1,
                        "pronunciationAccuracyScore": 101,
                    }
                ]
            }
        )
