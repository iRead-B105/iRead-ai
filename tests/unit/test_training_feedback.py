from __future__ import annotations

from iread_ai.training_feedback import build_pronunciation_feedback


def _analysis() -> dict[str, object]:
    return {
        "pronunciationAccuracyScore": 44,
        "fluencyScore": 82,
        "completenessScore": 0,
        "confidence": 0.956,
        "recognizedText": "토끼가 숲길을 천천히 걸어요.",
        "words": [
            {
                "word": "토끼가",
                "accuracyScore": 44,
                "errorType": "Mispronunciation",
                "offsetMs": 1590,
                "durationMs": 1380,
            },
            {
                "word": "숲길을",
                "accuracyScore": 42,
                "errorType": "Mispronunciation",
                "offsetMs": 3350,
                "durationMs": 1290,
            },
            {
                "word": "천천히",
                "accuracyScore": 35,
                "errorType": "Mispronunciation",
                "offsetMs": 5010,
                "durationMs": 1440,
            },
            {
                "word": "걸어요",
                "accuracyScore": 53,
                "errorType": "Mispronunciation",
                "offsetMs": 6890,
                "durationMs": 1260,
            },
        ],
    }


def test_sentence_repeat_feedback_uses_fluency_and_two_weakest_words() -> None:
    feedback = build_pronunciation_feedback(
        _analysis(),
        reference_text="토끼가 숲길을 천천히 걸어요.",
        training_template_id=30,
    )

    assert feedback.evaluation_focus == "단어별 정확도와 문장 유창성"
    assert feedback.focus_words == ("천천히", "숲길을")
    assert feedback.retry_recommended is True
    assert "읽는 흐름은 좋았어요" in feedback.child_summary
    assert "천천히 35점" in feedback.teacher_observation
    assert any("완성도는 0점" in caution for caution in feedback.cautions)


def test_omission_is_prioritized_without_inventing_a_phoneme_error() -> None:
    analysis = _analysis()
    words = list(analysis["words"])  # type: ignore[arg-type]
    words[1] = {
        "word": "숲길을",
        "accuracyScore": None,
        "errorType": "Omission",
        "offsetMs": 0,
        "durationMs": 0,
    }
    analysis["words"] = words

    feedback = build_pronunciation_feedback(
        analysis,
        reference_text="토끼가 숲길을 천천히 걸어요.",
        training_template_id=25,
    )

    assert feedback.focus_words[0] == "숲길을"
    assert feedback.words[1].label == "읽지 않음"
    assert "자음·모음 대치를 추정하지 않습니다" in feedback.teacher_observation


def test_good_word_reading_does_not_recommend_retry() -> None:
    analysis = {
        "pronunciationAccuracyScore": 91,
        "fluencyScore": 88,
        "completenessScore": 100,
        "confidence": 0.94,
        "recognizedText": "사과를 먹어요.",
        "words": [
            {
                "word": "사과를",
                "accuracyScore": 92,
                "errorType": "None",
                "offsetMs": 200,
                "durationMs": 700,
            },
            {
                "word": "먹어요",
                "accuracyScore": 90,
                "errorType": "None",
                "offsetMs": 1000,
                "durationMs": 800,
            },
        ],
    }

    feedback = build_pronunciation_feedback(
        analysis,
        reference_text="사과를 먹어요.",
        training_template_id=25,
    )

    assert feedback.focus_words == ()
    assert feedback.retry_recommended is False
    assert feedback.child_summary.startswith("잘 읽었어요")
