from __future__ import annotations

from iread_ai.training_personalization import select_training_candidate


def test_selects_candidate_that_uses_target_palette_and_avoids_excluded_feature() -> None:
    candidates = [
        {"sentences": ["꽃이 피어요."]},
        {"sentences": ["나비가 날아요."]},
        {"sentences": ["꿀이 달아요."]},
    ]

    selected, evidence = select_training_candidate(
        candidates,
        target_features=["GRAPHEME.ONSET.TENSE.ㄲ"],
        excluded_features=["GRAPHEME.CODA.SIMPLE.ㅊ"],
        recommended_words=["꽃", "꿀"],
        analyzer=None,
        lexicon_applied=True,
        training_type="SENTENCE_READING",
        difficulty=3,
    )

    assert selected == {"sentences": ["꿀이 달아요."]}
    assert evidence.selectedCandidateIndex == 2
    assert evidence.candidates[2].targetOccurrences["GRAPHEME.ONSET.TENSE.ㄲ"] == 1
    assert evidence.candidates[2].excludedOccurrences["GRAPHEME.CODA.SIMPLE.ㅊ"] == 0
    assert evidence.candidates[2].paletteWordUses == ["꿀"]


def test_candidate_selection_is_available_without_lexicon() -> None:
    selected, evidence = select_training_candidate(
        [{"sentence": "나비가 가요."}, {"sentence": "꿀이 와요."}],
        target_features=["GRAPHEME.ONSET.TENSE.ㄲ"],
        excluded_features=[],
        recommended_words=[],
        analyzer=None,
        lexicon_applied=False,
        training_type="SENTENCE_READING",
        difficulty=3,
    )

    assert selected == {"sentence": "꿀이 와요."}
    assert evidence.lexiconApplied is False
    assert evidence.selectedCandidateIndex == 1


def test_rule_based_targets_are_counted_from_structured_fields() -> None:
    _, consonant_evidence = select_training_candidate(
        [{"consonantType": "BASIC", "target": "ㄱ", "soundText": "기역"}],
        target_features=["GRAPHEME.ONSET.BASIC.ㄱ"],
        excluded_features=[],
        recommended_words=[],
        analyzer=None,
        lexicon_applied=False,
        training_type="CONSONANT_TRACE",
        difficulty=1,
    )
    _, syllable_evidence = select_training_candidate(
        [{"syllableType": "WITHOUT_FINAL", "target": "가", "soundText": "가"}],
        target_features=["SYLLABLE.CV"],
        excluded_features=[],
        recommended_words=[],
        analyzer=None,
        lexicon_applied=False,
        training_type="SYLLABLE_TRACE",
        difficulty=1,
    )
    _, word_evidence = select_training_candidate(
        [{"readingOrder": "SEQUENTIAL", "words": ["나무", "라디오", "바다"]}],
        target_features=["WORD.SYLLABLE_COUNT.2"],
        excluded_features=[],
        recommended_words=[],
        analyzer=None,
        lexicon_applied=False,
        training_type="WORD_READING",
        difficulty=3,
    )

    consonant_fit = consonant_evidence.candidates[0]
    syllable_fit = syllable_evidence.candidates[0]
    word_fit = word_evidence.candidates[0]
    assert consonant_fit.targetOccurrences == {"GRAPHEME.ONSET.BASIC.ㄱ": 1}
    assert consonant_fit.analysisStatus == "STRUCTURED_VERIFIED"
    assert syllable_fit.targetOccurrences == {"SYLLABLE.CV": 1}
    assert word_fit.targetOccurrences == {"WORD.SYLLABLE_COUNT.2": 2}


def test_prefers_child_readable_length_and_penalizes_target_overuse() -> None:
    candidates = [
        {
            "title": "긴 이야기",
            "sentences": [
                {
                    "speaker": "NARRATOR",
                    "text": "까치가 까만 깃털을 펴고 깨끗한 뜰을 아주 빠르게 빙글빙글 돌았어요.",
                    "emotion": "EXCITED",
                },
                {
                    "speaker": "CHARACTER",
                    "text": "까까까 소리를 내며 토끼 곁에서 껑충껑충 뛰었어요.",
                    "emotion": "HAPPY",
                },
            ],
        },
        {
            "title": "알맞은 이야기",
            "sentences": [
                {
                    "speaker": "NARRATOR",
                    "text": "토끼가 작은 상자를 꺼냈어요.",
                    "emotion": "CALM",
                },
                {
                    "speaker": "CHARACTER",
                    "text": "친구에게 꼭 보여 주고 싶어.",
                    "emotion": "HAPPY",
                },
                {
                    "speaker": "NARRATOR",
                    "text": "둘은 상자를 함께 열었어요.",
                    "emotion": "EXCITED",
                },
                {
                    "speaker": "NARRATOR",
                    "text": "상자 안에서 작은 별이 나왔어요.",
                    "emotion": "SURPRISED",
                },
            ],
        },
    ]

    selected, evidence = select_training_candidate(
        candidates,
        target_features=["GRAPHEME.ONSET.TENSE.ㄲ"],
        excluded_features=[],
        recommended_words=["토끼", "꺼내다"],
        analyzer=None,
        lexicon_applied=True,
        training_type="SHORT_STORY_READING",
        difficulty=3,
    )

    assert selected["title"] == "알맞은 이야기"
    assert evidence.candidates[0].lengthStatus == "TOO_LONG"
    assert evidence.candidates[1].lengthStatus == "PASS"
    assert evidence.candidates[1].sentenceSyllableCounts == [12, 11, 11, 13]


def test_length_pass_has_priority_over_a_higher_raw_score() -> None:
    long_candidate = {
        "title": "목표가 너무 많은 글",
        "sentences": [
            {
                "speaker": "NARRATOR",
                "text": "까치가 까만 깃털을 펴고 깨끗한 뜰에서 껑충껑충 크게 뛰었어요.",
                "emotion": "EXCITED",
            },
            {
                "speaker": "CHARACTER",
                "text": "토끼도 깜짝 놀라 까치 곁으로 아주 빠르게 달려갔어요.",
                "emotion": "SURPRISED",
            },
            {
                "speaker": "NARRATOR",
                "text": "둘은 끝까지 까르르 웃으며 꽃밭을 오래오래 뛰어다녔어요.",
                "emotion": "HAPPY",
            },
        ],
    }
    readable_candidate = {
        "title": "읽기 좋은 글",
        "sentences": [
            {"speaker": "NARRATOR", "text": "토끼가 작은 상자를 꺼냈어요.", "emotion": "CALM"},
            {"speaker": "CHARACTER", "text": "친구에게 꼭 보여 주고 싶어.", "emotion": "HAPPY"},
            {"speaker": "NARRATOR", "text": "둘은 상자를 함께 열었어요.", "emotion": "EXCITED"},
            {
                "speaker": "NARRATOR",
                "text": "상자 안에서 작은 별이 나왔어요.",
                "emotion": "SURPRISED",
            },
        ],
    }

    selected, evidence = select_training_candidate(
        [long_candidate, readable_candidate],
        target_features=["GRAPHEME.ONSET.TENSE.ㄲ"],
        excluded_features=[],
        recommended_words=[],
        analyzer=None,
        lexicon_applied=False,
        training_type="SHORT_STORY_READING",
        difficulty=3,
    )

    assert evidence.candidates[0].lengthStatus == "TOO_LONG"
    assert evidence.candidates[1].lengthStatus == "PASS"
    assert selected["title"] == "읽기 좋은 글"
