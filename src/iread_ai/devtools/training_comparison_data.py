from __future__ import annotations

from typing import Any

TRAINING_TYPES = (
    (1, "글자 따라 보기", "모음 따라 보기", "VOWEL_TRACE"),
    (2, "글자 따라 보기", "자음 따라 보기", "CONSONANT_TRACE"),
    (3, "글자 따라 보기", "음절 따라 보기", "SYLLABLE_TRACE"),
    (4, "소리 듣고 고르기", "자음 소리 고르기", "CONSONANT_SOUND_CHOICE"),
    (5, "소리 듣고 고르기", "모음 소리 고르기", "VOWEL_SOUND_CHOICE"),
    (6, "소리 듣고 고르기", "자음·모음 구별하기", "CONSONANT_VOWEL_CLASSIFICATION"),
    (7, "소리 듣고 고르기", "음절의 첫소리 찾기", "SYLLABLE_INITIAL_CHOICE"),
    (8, "소리 듣고 고르기", "낱말의 첫소리 찾기", "WORD_INITIAL_CHOICE"),
    (9, "소리 듣고 고르기", "같은 첫소리 낱말 찾기", "SAME_INITIAL_WORD_CHOICE"),
    (10, "소리 듣고 고르기", "받침 소리 고르기", "FINAL_CONSONANT_CHOICE"),
    (11, "소리 듣고 고르기", "낱말의 끝소리 고르기", "WORD_FINAL_SOUND_CHOICE"),
    (12, "소리 듣고 고르기", "서로 다른 받침 음절 비교하기", "FINAL_CONSONANT_COMPARISON"),
    (13, "소리 듣고 고르기", "비슷한 소리 고르기", "SIMILAR_SOUND_CHOICE"),
    (14, "글자 만들기", "음소 합쳐 음절 만들기", "PHONEME_BLEND"),
    (15, "글자 만들기", "음절 합쳐 낱말 만들기", "SYLLABLE_BLEND"),
    (16, "글자 만들기", "기본 글자 만들기", "BASIC_SYLLABLE_BUILD"),
    (17, "글자 만들기", "받침 글자 만들기", "FINAL_SYLLABLE_BUILD"),
    (18, "글자 만들기", "겹받침 글자 만들기", "DOUBLE_FINAL_BUILD"),
    (19, "글자 자르기", "받침 빼기", "FINAL_CONSONANT_DELETE"),
    (20, "글자 자르기", "음절 빼기", "SYLLABLE_DELETE"),
    (21, "글자 대치", "음절 바꾸기", "SYLLABLE_REPLACE"),
    (22, "글 해독", "낱말 읽기", "WORD_READING"),
    (23, "글 해독", "새 낱말 읽기", "NONWORD_READING"),
    (24, "글 해독", "어려운 단어 먼저 읽기", "DIFFICULT_WORD_PREVIEW"),
    (25, "글 해독", "문장 읽기", "SENTENCE_READING"),
    (26, "글 해독", "짧은 글 읽기", "SHORT_PASSAGE_READING"),
    (27, "문장 완성 및 이해", "문장 전체 조립", "SENTENCE_ASSEMBLY"),
    (28, "문장 완성 및 이해", "빈칸에 알맞은 단어 넣기", "FILL_IN_THE_BLANK"),
    (29, "문장 완성 및 이해", "그림과 문장 연결하기", "IMAGE_SENTENCE_MATCH"),
    (30, "유창하게 읽기", "문장 따라 읽기", "SENTENCE_REPEAT"),
    (31, "유창하게 읽기", "단어 이어 읽기", "WORD_CHAIN_READING"),
    (32, "유창하게 읽기", "끊어 읽기", "PHRASE_READING"),
    (33, "유창하게 읽기", "같은 문장 다시 읽기", "REPEATED_SENTENCE_READING"),
    (34, "유창하게 읽기", "짧은 이야기 읽기", "SHORT_STORY_READING"),
)


AI_BEGINNER_PROFILE = (
    ("GRAPHEME.ONSET.ASPIRATED.ㅍ", 0.705, 0.82, 10),
    ("GRAPHEME.ONSET.TENSE.ㅆ", 0.705, 0.88, 11),
    ("GRAPHEME.ONSET.ASPIRATED.ㅎ", 0.697, 0.84, 11),
    ("GRAPHEME.ONSET.TENSE.ㅉ", 0.697, 0.82, 12),
    ("GRAPHEME.ONSET.ASPIRATED.ㅊ", 0.689, 0.84, 13),
    ("GRAPHEME.ONSET.TENSE.ㄲ", 0.689, 0.82, 8),
    ("GRAPHEME.ONSET.ASPIRATED.ㅋ", 0.681, 0.86, 8),
    ("GRAPHEME.ONSET.TENSE.ㄸ", 0.681, 0.84, 9),
    ("GRAPHEME.ONSET.ASPIRATED.ㅌ", 0.673, 0.88, 9),
    ("GRAPHEME.ONSET.TENSE.ㅃ", 0.673, 0.86, 10),
    ("WORD.SYLLABLE_COUNT.2", 0.669, 0.82, 10),
    ("WORD.SYLLABLE_COUNT.1", 0.622, 0.88, 9),
    ("GRAPHEME.ONSET.BASIC.ㄷ", 0.600, 0.84, 13),
    ("GRAPHEME.ONSET.BASIC.ㅇ", 0.600, 0.86, 12),
    ("GRAPHEME.VOWEL.BASIC.ㅑ", 0.600, 0.84, 9),
    ("GRAPHEME.VOWEL.BASIC.ㅖ", 0.600, 0.86, 8),
    ("GRAPHEME.VOWEL.BASIC.ㅡ", 0.600, 0.88, 13),
    ("SYLLABLE.CV", 0.597, 0.84, 9),
    ("GRAPHEME.ONSET.BASIC.ㄹ", 0.592, 0.86, 8),
    ("GRAPHEME.ONSET.BASIC.ㅈ", 0.592, 0.88, 13),
    ("GRAPHEME.VOWEL.BASIC.ㅒ", 0.592, 0.86, 10),
    ("GRAPHEME.VOWEL.BASIC.ㅗ", 0.592, 0.88, 9),
    ("GRAPHEME.VOWEL.BASIC.ㅣ", 0.592, 0.82, 8),
    ("GRAPHEME.ONSET.BASIC.ㅁ", 0.584, 0.88, 9),
    ("GRAPHEME.VOWEL.BASIC.ㅓ", 0.584, 0.88, 11),
    ("GRAPHEME.VOWEL.BASIC.ㅛ", 0.584, 0.82, 10),
    ("GRAPHEME.ONSET.BASIC.ㄱ", 0.576, 0.88, 11),
    ("GRAPHEME.ONSET.BASIC.ㅂ", 0.576, 0.82, 10),
    ("GRAPHEME.VOWEL.BASIC.ㅏ", 0.576, 0.88, 13),
    ("GRAPHEME.VOWEL.BASIC.ㅔ", 0.576, 0.82, 12),
    ("GRAPHEME.VOWEL.BASIC.ㅜ", 0.576, 0.84, 11),
    ("GRAPHEME.ONSET.BASIC.ㄴ", 0.568, 0.82, 12),
    ("GRAPHEME.ONSET.BASIC.ㅅ", 0.568, 0.84, 11),
    ("GRAPHEME.VOWEL.BASIC.ㅐ", 0.568, 0.82, 8),
    ("GRAPHEME.VOWEL.BASIC.ㅕ", 0.568, 0.84, 13),
    ("GRAPHEME.VOWEL.BASIC.ㅠ", 0.568, 0.86, 12),
)


SUPPORTED_PREFIXES = {
    "VOWEL_TRACE": ("GRAPHEME.VOWEL.",),
    "VOWEL_SOUND_CHOICE": ("GRAPHEME.VOWEL.",),
    "CONSONANT_TRACE": ("GRAPHEME.ONSET.",),
    "CONSONANT_SOUND_CHOICE": ("GRAPHEME.ONSET.",),
    "CONSONANT_VOWEL_CLASSIFICATION": ("GRAPHEME.ONSET.", "GRAPHEME.VOWEL."),
    "SYLLABLE_TRACE": ("SYLLABLE.",),
    "SYLLABLE_INITIAL_CHOICE": ("GRAPHEME.ONSET.",),
    "WORD_INITIAL_CHOICE": ("GRAPHEME.ONSET.",),
    "SAME_INITIAL_WORD_CHOICE": ("GRAPHEME.ONSET.",),
    "SIMILAR_SOUND_CHOICE": ("GRAPHEME.ONSET.",),
    "FINAL_CONSONANT_CHOICE": ("GRAPHEME.CODA.",),
    "FINAL_CONSONANT_COMPARISON": ("GRAPHEME.CODA.",),
    "FINAL_CONSONANT_DELETE": ("GRAPHEME.CODA.",),
    "WORD_FINAL_SOUND_CHOICE": ("GRAPHEME.CODA.", "PHONOLOGY.FINAL_NEUTRALIZATION"),
    "PHONEME_BLEND": ("SYLLABLE.", "GRAPHEME."),
    "SYLLABLE_BLEND": ("WORD.", "SYLLABLE."),
    "BASIC_SYLLABLE_BUILD": ("SYLLABLE.CV", "GRAPHEME.ONSET.", "GRAPHEME.VOWEL."),
    "FINAL_SYLLABLE_BUILD": (
        "SYLLABLE.CVC",
        "GRAPHEME.CODA.SIMPLE.",
        "GRAPHEME.ONSET.",
        "GRAPHEME.VOWEL.",
    ),
    "DOUBLE_FINAL_BUILD": ("SYLLABLE.COMPLEX_CODA", "GRAPHEME.CODA.COMPLEX."),
    "SYLLABLE_DELETE": ("WORD.", "SYLLABLE."),
    "SYLLABLE_REPLACE": ("WORD.", "SYLLABLE."),
    "WORD_READING": ("WORD.", "GRAPHEME.", "PHONOLOGY."),
    "NONWORD_READING": ("WORD.", "GRAPHEME.", "PHONOLOGY."),
    "WORD_CHAIN_READING": ("WORD.", "GRAPHEME.", "PHONOLOGY."),
    "DIFFICULT_WORD_PREVIEW": ("WORD.", "PHONOLOGY."),
    "SENTENCE_READING": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "SHORT_PASSAGE_READING": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "SENTENCE_ASSEMBLY": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "FILL_IN_THE_BLANK": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "IMAGE_SENTENCE_MATCH": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "SENTENCE_REPEAT": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "PHRASE_READING": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "REPEATED_SENTENCE_READING": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
    "SHORT_STORY_READING": ("SENTENCE.", "WORD.", "PHONOLOGY.", "GRAPHEME."),
}


ORIGINAL_MOCKS: dict[str, dict[str, Any]] = {
    "VOWEL_TRACE": {
        "content": {"target": "ㅏ", "soundText": "ㅏ", "traceAssetKey": "vowel_0"},
        "answer": {"target": "ㅏ"},
    },
    "CONSONANT_TRACE": {
        "content": {"target": "ㄱ", "soundText": "ㄱ", "traceAssetKey": "consonant_0"},
        "answer": {"target": "ㄱ"},
    },
    "SYLLABLE_TRACE": {
        "content": {"target": "가", "soundText": "가", "traceAssetKey": "syllable_0"},
        "answer": {"target": "가"},
    },
    "CONSONANT_SOUND_CHOICE": {
        "content": {"audioText": "ㄱ", "choices": ["ㄱ", "ㄴ", "ㄷ"]},
        "answer": {"answerIndex": 0},
    },
    "VOWEL_SOUND_CHOICE": {
        "content": {"audioText": "ㅏ", "choices": ["ㅏ", "ㅓ", "ㅗ"]},
        "answer": {"answerIndex": 0},
    },
    "CONSONANT_VOWEL_CLASSIFICATION": {
        "content": {"audioText": "ㄱ", "choices": ["CONSONANT", "VOWEL"]},
        "answer": {"answerIndex": 0},
    },
    "SYLLABLE_INITIAL_CHOICE": {
        "content": {"audioText": "가", "choices": ["ㄱ", "ㄴ", "ㄷ"]},
        "answer": {"answerIndex": 0},
    },
    "WORD_INITIAL_CHOICE": {
        "content": {"audioText": "사과", "choices": ["ㅅ", "ㄱ", "ㄴ"]},
        "answer": {"answerIndex": 0},
    },
    "SAME_INITIAL_WORD_CHOICE": {
        "content": {
            "targetAudioText": "사과",
            "choiceType": "WORD",
            "choices": [{"text": "수박"}, {"text": "기차"}, {"text": "연필"}],
        },
        "answer": {"answerIndex": 0},
    },
    "FINAL_CONSONANT_CHOICE": {
        "content": {"audioText": "각", "choices": ["ㄱ", "ㄴ", "ㄹ"]},
        "answer": {"answerIndex": 0},
    },
    "WORD_FINAL_SOUND_CHOICE": {
        "content": {"audioText": "산", "choices": ["ㄴ", "ㄱ", "ㅁ"]},
        "answer": {"answerIndex": 0},
    },
    "FINAL_CONSONANT_COMPARISON": {
        "content": {"audioText": "각", "choices": ["각", "간", "갈"]},
        "answer": {"answerIndex": 0},
    },
    "SIMILAR_SOUND_CHOICE": {
        "content": {
            "soundGroup": "평음·격음·경음",
            "audioText": "가",
            "choices": ["가", "카", "까"],
        },
        "answer": {"answerIndex": 0},
    },
    "PHONEME_BLEND": {
        "content": {"audioParts": ["ㄱ", "ㅏ"], "cards": ["ㄱ", "ㅏ", "ㄴ"]},
        "answer": {"answerOrder": [0, 1], "result": "가"},
    },
    "SYLLABLE_BLEND": {
        "content": {"audioParts": ["사", "과"], "cards": ["사", "과", "나"]},
        "answer": {"answerOrder": [0, 1], "result": "사과"},
    },
    "BASIC_SYLLABLE_BUILD": {
        "content": {
            "targetAudioText": "가",
            "initialChoices": ["ㄱ", "ㄴ"],
            "medialChoices": ["ㅏ", "ㅓ"],
        },
        "answer": {"initialAnswerIndex": 0, "medialAnswerIndex": 0, "result": "가"},
    },
    "FINAL_SYLLABLE_BUILD": {
        "content": {
            "targetAudioText": "각",
            "initialChoices": ["ㄱ", "ㄴ"],
            "medialChoices": ["ㅏ", "ㅓ"],
            "finalChoices": ["ㄱ", "ㄴ"],
        },
        "answer": {
            "initialAnswerIndex": 0,
            "medialAnswerIndex": 0,
            "finalAnswerIndex": 0,
            "result": "각",
        },
    },
    "DOUBLE_FINAL_BUILD": {
        "content": {
            "targetAudioText": "닭",
            "initialChoices": ["ㄷ", "ㄱ"],
            "medialChoices": ["ㅏ", "ㅓ"],
            "finalChoices": ["ㄺ", "ㄱ"],
        },
        "answer": {
            "initialAnswerIndex": 0,
            "medialAnswerIndex": 0,
            "finalAnswerIndex": 0,
            "result": "닭",
        },
    },
    "FINAL_CONSONANT_DELETE": {
        "content": {"source": "감", "targetAudioText": "가", "removableUnits": ["ㄱ", "ㅏ", "ㅁ"]},
        "answer": {"answerIndex": 2, "result": "가"},
    },
    "SYLLABLE_DELETE": {
        "content": {"source": "사과", "targetAudioText": "과", "syllables": ["사", "과"]},
        "answer": {"deleteIndex": 0, "result": "과"},
    },
    "SYLLABLE_REPLACE": {
        "content": {
            "source": "사과",
            "targetAudioText": "나과",
            "replaceIndex": 0,
            "choices": ["나", "다"],
        },
        "answer": {"replaceIndex": 0, "answerIndex": 0, "result": "나과"},
    },
    "WORD_READING": {
        "content": {"words": ["사과", "나무", "바다"]},
        "answer": {"expectedText": "사과 나무 바다"},
    },
    "NONWORD_READING": {
        "content": {"words": [{"text": "나무"}, {"text": "두미"}]},
        "answer": {"expectedText": "나무 두미"},
    },
    "DIFFICULT_WORD_PREVIEW": {
        "content": {
            "difficultWords": [{"word": "사과", "syllables": ["사", "과"]}],
            "sentence": "아기는 사과를 먹는다.",
        },
        "answer": {"expectedText": "아기는 사과를 먹는다."},
    },
    "SENTENCE_READING": {
        "content": {"sentence": "아기는 사과를 먹는다.", "tokens": ["아기는", "사과를", "먹는다."]},
        "answer": {"expectedText": "아기는 사과를 먹는다."},
    },
    "SHORT_PASSAGE_READING": {
        "content": {"sentences": ["아기는 사과를 먹는다.", "나무 위에서 새가 노래한다."]},
        "answer": {"expectedText": "아기는 사과를 먹는다. 나무 위에서 새가 노래한다."},
    },
    "SENTENCE_ASSEMBLY": {
        "content": {"cards": ["사과를", "먹는다.", "아기는"]},
        "answer": {"answerOrder": [2, 0, 1], "completedSentence": "아기는 사과를 먹는다."},
    },
    "FILL_IN_THE_BLANK": {
        "content": {
            "sentence": "책상 위에 {{blank}} 그림이 있다.",
            "inputType": "CHOICE",
            "choices": ["사과", "기차", "연필"],
        },
        "answer": {"answerIndex": 0, "completedSentence": "책상 위에 사과 그림이 있다."},
    },
    "IMAGE_SENTENCE_MATCH": {
        "content": {
            "imagePrompt": "아기가 사과를 먹는 장면",
            "imageUrl": "",
            "choices": ["아기는 사과를 먹는다.", "비가 내린다."],
        },
        "answer": {"answerIndex": 0},
    },
    "SENTENCE_REPEAT": {
        "content": {"sentence": "아기는 사과를 먹는다.", "emotion": "HAPPY"},
        "answer": {"expectedText": "아기는 사과를 먹는다."},
    },
    "WORD_CHAIN_READING": {
        "content": {"words": ["사과", "나무", "바다"], "requiredOrder": "SEQUENTIAL"},
        "answer": {"expectedText": "사과 나무 바다"},
    },
    "PHRASE_READING": {
        "content": {"sentence": "아기는 사과를 먹는다.", "phrases": ["아기는", "사과를 먹는다."]},
        "answer": {"expectedText": "아기는 사과를 먹는다."},
    },
    "REPEATED_SENTENCE_READING": {
        "content": {"sentence": "아기는 사과를 먹는다.", "repeatCount": 2},
        "answer": {"expectedText": "아기는 사과를 먹는다."},
    },
    "SHORT_STORY_READING": {
        "content": {
            "title": "사과 이야기",
            "sentences": [
                {"speaker": "NARRATOR", "text": "아기는 사과를 먹는다."},
                {"speaker": "CHARACTER", "text": "정말 맛있어!"},
            ],
        },
        "answer": {"expectedText": "아기는 사과를 먹는다. 정말 맛있어!"},
    },
}


def beginner_targets(training_type: str) -> list[dict[str, Any]]:
    prefixes = SUPPORTED_PREFIXES.get(training_type, ())
    matched = [
        row for row in AI_BEGINNER_PROFILE if any(row[0].startswith(prefix) for prefix in prefixes)
    ]
    selected: list[tuple[str, float, float, int]] = []
    families: set[str] = set()
    for row in matched:
        family = ".".join(row[0].split(".")[:3])
        if family in families and len(matched) > 1:
            continue
        selected.append(row)
        families.add(family)
        if len(selected) == 2:
            break
    for row in matched:
        if row not in selected:
            selected.append(row)
        if len(selected) == 2:
            break
    return [
        {
            "featureCode": code,
            "weaknessScore": weakness,
            "confidence": confidence,
            "evidenceCount": evidence,
        }
        for code, weakness, confidence, evidence in selected
    ]


__all__ = [
    "AI_BEGINNER_PROFILE",
    "ORIGINAL_MOCKS",
    "SUPPORTED_PREFIXES",
    "TRAINING_TYPES",
    "beginner_targets",
]
