from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iread_ai.generation_models import TrainingCandidateRequest
from iread_ai.mock_generators import generate_training_candidates


@dataclass(frozen=True, slots=True)
class TrainingReviewSpec:
    template_id: int
    group: str
    name: str
    training_type: str
    strategy: str
    suggested_feature: str


def _spec(
    template_id: int,
    group: str,
    name: str,
    training_type: str,
    suggested_feature: str,
) -> TrainingReviewSpec:
    strategy = "규칙·사전" if template_id <= 23 or template_id == 31 else "LLM+로컬 검증"
    return TrainingReviewSpec(
        template_id,
        group,
        name,
        training_type,
        strategy,
        suggested_feature,
    )


TRAINING_REVIEW_CATALOG = (
    _spec(1, "글자 따라 보기", "모음 따라 보기", "VOWEL_TRACE", "GRAPHEME.VOWEL.BASIC.ㅏ"),
    _spec(2, "글자 따라 보기", "자음 따라 보기", "CONSONANT_TRACE", "GRAPHEME.ONSET.BASIC.ㄱ"),
    _spec(3, "글자 따라 보기", "음절 따라 보기", "SYLLABLE_TRACE", "SYLLABLE.CV"),
    _spec(
        4,
        "소리 듣고 고르기",
        "자음 소리 고르기",
        "CONSONANT_SOUND_CHOICE",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(
        5, "소리 듣고 고르기", "모음 소리 고르기", "VOWEL_SOUND_CHOICE", "GRAPHEME.VOWEL.BASIC.ㅏ"
    ),
    _spec(
        6,
        "소리 듣고 고르기",
        "자음·모음 구별하기",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(
        7,
        "소리 듣고 고르기",
        "음절의 첫소리 찾기",
        "SYLLABLE_INITIAL_CHOICE",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(
        8,
        "소리 듣고 고르기",
        "낱말의 첫소리 찾기",
        "WORD_INITIAL_CHOICE",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(
        9,
        "소리 듣고 고르기",
        "같은 첫소리 낱말 찾기",
        "SAME_INITIAL_WORD_CHOICE",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(
        10,
        "소리 듣고 고르기",
        "받침 소리 고르기",
        "FINAL_CONSONANT_CHOICE",
        "GRAPHEME.CODA.SIMPLE.ㄴ",
    ),
    _spec(
        11,
        "소리 듣고 고르기",
        "낱말의 끝소리 고르기",
        "WORD_FINAL_SOUND_CHOICE",
        "GRAPHEME.CODA.SIMPLE.ㄴ",
    ),
    _spec(
        12,
        "소리 듣고 고르기",
        "서로 다른 받침 음절 비교하기",
        "FINAL_CONSONANT_COMPARISON",
        "GRAPHEME.CODA.SIMPLE.ㄴ",
    ),
    _spec(
        13,
        "소리 듣고 고르기",
        "비슷한 소리 고르기",
        "SIMILAR_SOUND_CHOICE",
        "GRAPHEME.ONSET.BASIC.ㄱ",
    ),
    _spec(14, "글자 만들기", "음소 합쳐 음절 만들기", "PHONEME_BLEND", "SYLLABLE.CV"),
    _spec(15, "글자 만들기", "음절 합쳐 낱말 만들기", "SYLLABLE_BLEND", "WORD.SYLLABLE_COUNT.2"),
    _spec(16, "글자 만들기", "기본 글자 만들기", "BASIC_SYLLABLE_BUILD", "SYLLABLE.CV"),
    _spec(17, "글자 만들기", "받침 글자 만들기", "FINAL_SYLLABLE_BUILD", "SYLLABLE.CVC"),
    _spec(18, "글자 만들기", "겹받침 글자 만들기", "DOUBLE_FINAL_BUILD", "SYLLABLE.COMPLEX_CODA"),
    _spec(19, "글자 자르기", "받침 빼기", "FINAL_CONSONANT_DELETE", "SYLLABLE.CVC"),
    _spec(20, "글자 자르기", "음절 빼기", "SYLLABLE_DELETE", "WORD.SYLLABLE_COUNT.2"),
    _spec(21, "글자 대치", "음절 바꾸기", "SYLLABLE_REPLACE", "WORD.SYLLABLE_COUNT.2"),
    _spec(22, "글 해독", "낱말 읽기", "WORD_READING", "WORD.SYLLABLE_COUNT.2"),
    _spec(23, "글 해독", "새 낱말 읽기", "NONWORD_READING", "WORD.SYLLABLE_COUNT.2"),
    _spec(
        24, "글 해독", "어려운 단어 먼저 읽기", "DIFFICULT_WORD_PREVIEW", "PHONOLOGY.NASALIZATION"
    ),
    _spec(25, "글 해독", "문장 읽기", "SENTENCE_READING", "PHONOLOGY.NASALIZATION"),
    _spec(26, "글 해독", "짧은 글 읽기", "SHORT_PASSAGE_READING", "PHONOLOGY.NASALIZATION"),
    _spec(27, "문장 완성 및 이해", "문장 전체 조립", "SENTENCE_ASSEMBLY", "SENTENCE.SIMPLE"),
    _spec(
        28,
        "문장 완성 및 이해",
        "빈칸에 알맞은 단어 넣기",
        "FILL_IN_THE_BLANK",
        "WORD.SYLLABLE_COUNT.2",
    ),
    _spec(
        29, "문장 완성 및 이해", "그림과 문장 연결하기", "IMAGE_SENTENCE_MATCH", "SENTENCE.SIMPLE"
    ),
    _spec(30, "유창하게 읽기", "문장 따라 읽기", "SENTENCE_REPEAT", "PHONOLOGY.NASALIZATION"),
    _spec(31, "유창하게 읽기", "단어 이어 읽기", "WORD_CHAIN_READING", "WORD.SYLLABLE_COUNT.2"),
    _spec(32, "유창하게 읽기", "끊어 읽기", "PHRASE_READING", "PHONOLOGY.NASALIZATION"),
    _spec(
        33,
        "유창하게 읽기",
        "같은 문장 다시 읽기",
        "REPEATED_SENTENCE_READING",
        "PHONOLOGY.NASALIZATION",
    ),
    _spec(34, "유창하게 읽기", "짧은 이야기 읽기", "SHORT_STORY_READING", "PHONOLOGY.NASALIZATION"),
)


def output_template(training_type: str) -> dict[str, Any]:
    request = TrainingCandidateRequest(
        requestId=f"catalog-{training_type}",
        schemaVersion=2,
        trainingType=training_type,
        count=5,
        difficulty=2,
        targetFeatures=[],
        excludedFeatures=[],
        additionalPrompt="",
        outputTemplate={"type": training_type, "data": [{}]},
    )
    sample = generate_training_candidates(request).data[0]
    return {"type": training_type, "data": [_placeholder(sample)]}


def _placeholder(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _placeholder(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_placeholder(value[0])] if value else ["<string>"]
    if isinstance(value, bool):
        return "<boolean>"
    if isinstance(value, int):
        return "<integer>"
    if isinstance(value, float):
        return "<number>"
    return "<string>"


__all__ = ["TRAINING_REVIEW_CATALOG", "TrainingReviewSpec", "output_template"]
