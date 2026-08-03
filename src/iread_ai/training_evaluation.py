from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .generation_models import EvaluateTrainingRequest, EvaluateTrainingResponse

EVALUATION_VERSION = "IREAD_TRAINING_EVALUATION_V1"
_MAX_PERCENT_SCORE = 100.0
_MAX_SCALED_SCORE = 1000.0


class TrainingEvaluationError(ValueError):
    """Raised when a training result contains an invalid scoring value."""


@dataclass(frozen=True, slots=True)
class QuestionScore:
    question_key: str
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class TrainingEvaluationBreakdown:
    evaluation_version: str
    accuracy: float
    question_scores: tuple[QuestionScore, ...]
    warnings: tuple[str, ...]


def evaluate_training(request: EvaluateTrainingRequest) -> EvaluateTrainingResponse:
    breakdown = evaluate_training_result(request.result)
    return EvaluateTrainingResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        accuracy=breakdown.accuracy,
    )


def evaluate_training_result(result: Mapping[str, Any]) -> TrainingEvaluationBreakdown:
    scores: dict[str, QuestionScore] = {}
    warnings: list[str] = []

    questions = _optional_sequence(result, "questions")
    for index, raw_question in enumerate(questions):
        question = _mapping(raw_question, f"questions[{index}]")
        score = _objective_score(question, f"questions[{index}]")
        if score is None:
            continue
        key = _question_key(question, fallback=f"objective:{index + 1}")
        scores[key] = QuestionScore(key, "OBJECTIVE", score)

    latest_pronunciations = _latest_pronunciation_by_question(
        _optional_sequence(result, "pronunciationAnalyses")
    )
    for key, analysis in latest_pronunciations.items():
        score = _percentage(
            analysis.get("pronunciationAccuracyScore"),
            f"pronunciationAnalyses[{key}].pronunciationAccuracyScore",
        )
        if score is None:
            continue
        scores[key] = QuestionScore(key, "PRONUNCIATION", score)

    if not scores:
        word_attempt_scores = _final_word_attempt_scores(_optional_sequence(result, "wordAttempts"))
        for index, score in enumerate(word_attempt_scores, start=1):
            key = f"word-attempt:{index}"
            scores[key] = QuestionScore(key, "WORD_ATTEMPT", score)

    if not scores:
        warnings.append("NO_SCORABLE_EVIDENCE")
        accuracy = 0.0
    else:
        accuracy = round(
            sum(item.score for item in scores.values()) / len(scores),
            2,
        )

    return TrainingEvaluationBreakdown(
        evaluation_version=EVALUATION_VERSION,
        accuracy=accuracy,
        question_scores=tuple(scores.values()),
        warnings=tuple(warnings),
    )


def _objective_score(question: Mapping[str, Any], path: str) -> float | None:
    if "totalScore" in question and question["totalScore"] is not None:
        return _scaled_score(question["totalScore"], f"{path}.totalScore")
    if question.get("isCorrect") is True or question.get("correctionConfirmed") is True:
        return 100.0
    if question.get("isCorrect") is False:
        return 0.0
    return None


def _latest_pronunciation_by_question(
    analyses: Sequence[Any],
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, tuple[int, int, Mapping[str, Any]]] = {}
    for index, raw_analysis in enumerate(analyses):
        analysis = _mapping(raw_analysis, f"pronunciationAnalyses[{index}]")
        key = _question_key(analysis, fallback=f"pronunciation:{index + 1}")
        attempt = _nonnegative_integer(
            analysis.get("attemptNo", index + 1),
            f"pronunciationAnalyses[{index}].attemptNo",
        )
        current = latest.get(key)
        if current is None or (attempt, index) >= (current[0], current[1]):
            latest[key] = (attempt, index, analysis)
    return {key: value[2] for key, value in latest.items()}


def _final_word_attempt_scores(attempts: Sequence[Any]) -> list[float]:
    scores: list[float] = []
    for index, raw_attempt in enumerate(attempts):
        attempt = _mapping(raw_attempt, f"wordAttempts[{index}]")
        if attempt.get("isFinal") is False:
            continue
        if "totalScore" in attempt and attempt["totalScore"] is not None:
            score = _scaled_score(attempt["totalScore"], f"wordAttempts[{index}].totalScore")
        else:
            score = _percentage(
                attempt.get("pronunciationAccuracyScore"),
                f"wordAttempts[{index}].pronunciationAccuracyScore",
            )
        if score is not None:
            scores.append(score)
    return scores


def _question_key(item: Mapping[str, Any], *, fallback: str) -> str:
    raw = item.get("questionNo", item.get("questionNumber", item.get("questionId")))
    if raw is None or isinstance(raw, bool):
        return fallback
    text = str(raw).strip()
    return f"question:{text}" if text else fallback


def _optional_sequence(result: Mapping[str, Any], field: str) -> Sequence[Any]:
    value = result.get(field, ())
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TrainingEvaluationError(f"result.{field} must be an array")
    return value


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingEvaluationError(f"{path} must be an object")
    return value


def _percentage(value: Any, path: str) -> float | None:
    if value is None:
        return None
    number = _number(value, path)
    if not 0 <= number <= _MAX_PERCENT_SCORE:
        raise TrainingEvaluationError(f"{path} must be between 0 and 100")
    return number


def _scaled_score(value: Any, path: str) -> float:
    number = _number(value, path)
    if not 0 <= number <= _MAX_SCALED_SCORE:
        raise TrainingEvaluationError(f"{path} must be between 0 and 1000")
    return round(number / 10, 2)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingEvaluationError(f"{path} must be numeric")
    return float(value)


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingEvaluationError(f"{path} must be a non-negative integer")
    return value


__all__ = [
    "EVALUATION_VERSION",
    "QuestionScore",
    "TrainingEvaluationBreakdown",
    "TrainingEvaluationError",
    "evaluate_training",
    "evaluate_training_result",
]
