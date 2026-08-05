from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

WordFeedbackStatus = Literal["STRONG", "PRACTICE", "FOCUS", "OMISSION", "INSERTION"]

DEFAULT_PRONUNCIATION_THRESHOLD = 70.0
FOCUS_SCORE_MAX = 50.0
FLUENCY_THRESHOLD = 70.0
LOW_RECOGNITION_CONFIDENCE = 0.50
MAX_FOCUS_WORDS = 2

_WORD_PATTERN = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+")
_FLUENCY_TEMPLATES = frozenset({30, 32, 33, 34})
_REPEATED_READING_TEMPLATES = frozenset({33})


class TrainingFeedbackError(ValueError):
    """Raised when pronunciation evidence cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class PronunciationWordFeedback:
    word: str
    score: float | None
    error_type: str
    status: WordFeedbackStatus
    label: str
    guidance: str
    offset_ms: int
    duration_ms: int

    @property
    def end_ms(self) -> int:
        return self.offset_ms + self.duration_ms


@dataclass(frozen=True, slots=True)
class PronunciationTrainingFeedback:
    evaluation_focus: str
    child_summary: str
    teacher_observation: str
    retry_recommended: bool
    focus_words: tuple[str, ...]
    strengths: tuple[str, ...]
    cautions: tuple[str, ...]
    words: tuple[PronunciationWordFeedback, ...]


def build_pronunciation_feedback(
    analysis: Mapping[str, Any],
    *,
    reference_text: str,
    training_template_id: int,
    pronunciation_threshold: float = DEFAULT_PRONUNCIATION_THRESHOLD,
) -> PronunciationTrainingFeedback:
    """Turn Azure word evidence into grounded product feedback without phoneme guesses."""

    if not 0 <= pronunciation_threshold <= 100:
        raise TrainingFeedbackError("pronunciation_threshold must be between 0 and 100")

    accuracy = _required_score(
        analysis.get("pronunciationAccuracyScore"),
        "pronunciationAccuracyScore",
    )
    fluency = _optional_score(analysis.get("fluencyScore"), "fluencyScore")
    completeness = _optional_score(
        analysis.get("completenessScore"),
        "completenessScore",
    )
    confidence = _optional_unit_score(analysis.get("confidence"), "confidence")
    raw_words = analysis.get("words")
    if (
        isinstance(raw_words, (str, bytes, bytearray))
        or not isinstance(raw_words, Sequence)
        or not raw_words
    ):
        raise TrainingFeedbackError("words must be a non-empty array")

    words = tuple(
        _word_feedback(item, index=index, threshold=pronunciation_threshold)
        for index, item in enumerate(raw_words)
    )
    focus_candidates = sorted(
        (word for word in words if word.status in {"FOCUS", "PRACTICE", "OMISSION"}),
        key=_focus_sort_key,
    )
    focus_words = tuple(dict.fromkeys(word.word for word in focus_candidates))[:MAX_FOCUS_WORDS]

    strengths: list[str] = []
    cautions: list[str] = []
    if accuracy >= pronunciation_threshold:
        strengths.append("기준 문장을 비교적 정확하게 읽었습니다.")
    if training_template_id in _FLUENCY_TEMPLATES and fluency is not None:
        if fluency >= FLUENCY_THRESHOLD:
            strengths.append("단어 사이의 멈춤과 읽는 흐름은 비교적 안정적입니다.")
        else:
            cautions.append("문장 흐름 점수가 낮아 의미 단위로 천천히 다시 읽어볼 필요가 있습니다.")
    if training_template_id in _REPEATED_READING_TEMPLATES:
        cautions.append("같은 문장 다시 읽기는 이전 시도와의 변화도 함께 확인해야 합니다.")
    if confidence is not None and confidence < LOW_RECOGNITION_CONFIDENCE:
        cautions.append("음성 인식 신뢰도가 낮아 조용한 환경에서 다시 녹음하는 것이 좋습니다.")

    recognized_text = analysis.get("recognizedText")
    if (
        isinstance(recognized_text, str)
        and _tokens(recognized_text) == _tokens(reference_text)
        and completeness == 0
    ):
        cautions.append(
            "인식 문장은 기준 문장과 일치하지만 완성도는 0점이므로 원본 녹음과 "
            "Azure 원시 결과를 재확인해야 합니다."
        )
    if any(word.status == "INSERTION" for word in words):
        cautions.append("기준 문장에 없는 단어가 추가로 인식되었습니다.")

    retry_recommended = bool(focus_words) or accuracy < pronunciation_threshold
    if confidence is not None and confidence < LOW_RECOGNITION_CONFIDENCE:
        retry_recommended = True

    child_summary = _child_summary(
        accuracy=accuracy,
        fluency=fluency,
        training_template_id=training_template_id,
        focus_words=focus_words,
        retry_recommended=retry_recommended,
        pronunciation_threshold=pronunciation_threshold,
    )
    teacher_observation = _teacher_observation(
        accuracy=accuracy,
        fluency=fluency,
        completeness=completeness,
        focus_words=focus_words,
        words=words,
        training_template_id=training_template_id,
    )

    return PronunciationTrainingFeedback(
        evaluation_focus=_evaluation_focus(training_template_id),
        child_summary=child_summary,
        teacher_observation=teacher_observation,
        retry_recommended=retry_recommended,
        focus_words=focus_words,
        strengths=tuple(strengths),
        cautions=tuple(dict.fromkeys(cautions)),
        words=words,
    )


def _word_feedback(
    value: Any,
    *,
    index: int,
    threshold: float,
) -> PronunciationWordFeedback:
    if not isinstance(value, Mapping):
        raise TrainingFeedbackError(f"words[{index}] must be an object")
    word = str(value.get("word", "")).strip()
    if not word:
        raise TrainingFeedbackError(f"words[{index}].word must be non-empty")
    error_type = str(value.get("errorType", "None")).strip() or "None"
    score = _optional_score(value.get("accuracyScore"), f"words[{index}].accuracyScore")
    offset_ms = _nonnegative_integer(value.get("offsetMs", 0), f"words[{index}].offsetMs")
    duration_ms = _nonnegative_integer(
        value.get("durationMs", 0),
        f"words[{index}].durationMs",
    )

    normalized_error = error_type.casefold()
    if normalized_error == "omission":
        status: WordFeedbackStatus = "OMISSION"
        label = "읽지 않음"
        guidance = f"‘{word}’을 빠뜨리지 않고 다시 읽어보세요."
    elif normalized_error == "insertion":
        status = "INSERTION"
        label = "추가로 읽음"
        guidance = "기준 문장에 있는 단어만 순서대로 읽어보세요."
    elif score is not None and score >= threshold and normalized_error == "none":
        status = "STRONG"
        label = "잘 읽음"
        guidance = "현재 읽기를 유지해 보세요."
    elif score is not None and score < FOCUS_SCORE_MAX:
        status = "FOCUS"
        label = "우선 연습"
        guidance = f"‘{word}’의 모범 음성을 듣고 한 번 더 또박또박 읽어보세요."
    else:
        status = "PRACTICE"
        label = "조금 더 연습"
        guidance = f"‘{word}’을 천천히 다시 읽어보세요."

    return PronunciationWordFeedback(
        word=word,
        score=score,
        error_type=error_type,
        status=status,
        label=label,
        guidance=guidance,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
    )


def _child_summary(
    *,
    accuracy: float,
    fluency: float | None,
    training_template_id: int,
    focus_words: tuple[str, ...],
    retry_recommended: bool,
    pronunciation_threshold: float,
) -> str:
    if not retry_recommended:
        if training_template_id in _FLUENCY_TEMPLATES and fluency is not None:
            return "문장을 정확하고 자연스럽게 읽었어요. 지금 읽는 흐름을 이어가 볼까요?"
        return "잘 읽었어요. 지금처럼 또박또박 읽어보세요."

    if focus_words:
        quoted = "와 ".join(f"‘{word}’" for word in focus_words)
        if training_template_id in _FLUENCY_TEMPLATES and fluency is not None and fluency >= 70:
            return f"읽는 흐름은 좋았어요. {quoted}을 한 번 더 또박또박 읽어볼까요?"
        return f"{quoted}을 천천히 한 번 더 읽어볼까요?"
    if accuracy < pronunciation_threshold:
        return "모범 음성을 다시 듣고 문장을 천천히 한 번 더 읽어볼까요?"
    return "조용한 곳에서 마이크를 가까이 두고 한 번 더 읽어볼까요?"


def _teacher_observation(
    *,
    accuracy: float,
    fluency: float | None,
    completeness: float | None,
    focus_words: tuple[str, ...],
    words: tuple[PronunciationWordFeedback, ...],
    training_template_id: int,
) -> str:
    parts = [f"전체 발음 정확도는 {accuracy:g}점입니다."]
    if training_template_id in _FLUENCY_TEMPLATES and fluency is not None:
        parts.append(f"문장 유창성은 {fluency:g}점입니다.")
    if completeness is not None:
        parts.append(f"완성도는 {completeness:g}점입니다.")
    if focus_words:
        details = ", ".join(
            f"{word.word} {word.score:g}점" if word.score is not None else f"{word.word} 누락"
            for word in words
            if word.word in focus_words and word.status != "INSERTION"
        )
        parts.append(f"우선 확인할 단어는 {details}입니다.")
    parts.append("이 결과는 단어 수준 관찰 근거이며, 구체적인 자음·모음 대치를 추정하지 않습니다.")
    return " ".join(parts)


def _evaluation_focus(training_template_id: int) -> str:
    if training_template_id in {1, 2, 3}:
        return "자모·음절 음성 입력 확인(점수는 검증 자료로만 사용)"
    if training_template_id in {15, 16, 17, 18, 19, 20, 21}:
        return "조작 결과를 소리 내어 읽은 정확도"
    if training_template_id in {22, 23, 24, 25, 26, 27, 28, 29, 31}:
        return "단어별 읽기 정확도와 누락 여부"
    if training_template_id in _FLUENCY_TEMPLATES:
        return "단어별 정확도와 문장 유창성"
    return "훈련 수행 정확도"


def _focus_sort_key(word: PronunciationWordFeedback) -> tuple[int, float, int]:
    omission_priority = 0 if word.status == "OMISSION" else 1
    score = word.score if word.score is not None else -1.0
    return omission_priority, score, word.offset_ms


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in _WORD_PATTERN.findall(text))


def _required_score(value: Any, path: str) -> float:
    score = _optional_score(value, path)
    if score is None:
        raise TrainingFeedbackError(f"{path} is required")
    return score


def _optional_score(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingFeedbackError(f"{path} must be numeric")
    score = float(value)
    if not 0 <= score <= 100:
        raise TrainingFeedbackError(f"{path} must be between 0 and 100")
    return score


def _optional_unit_score(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingFeedbackError(f"{path} must be numeric")
    score = float(value)
    if not 0 <= score <= 1:
        raise TrainingFeedbackError(f"{path} must be between 0 and 1")
    return score


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingFeedbackError(f"{path} must be a non-negative integer")
    return value


__all__ = [
    "DEFAULT_PRONUNCIATION_THRESHOLD",
    "PronunciationTrainingFeedback",
    "PronunciationWordFeedback",
    "TrainingFeedbackError",
    "build_pronunciation_feedback",
]
