from iread_ai.personalization.hangul import normalize_dialogue_quotes
from iread_ai.personalization.repair_policy import has_exact_spoken_dialogue


def test_converts_straight_double_quotes_to_curly() -> None:
    assert (
        normalize_dialogue_quotes('"가자!" 하고 별이가 외쳤어요.')
        == "“가자!” 하고 별이가 외쳤어요."
    )


def test_adds_missing_end_punctuation_inside_dialogue() -> None:
    assert (
        normalize_dialogue_quotes("“가자” 하고 별이가 말했어요.")
        == "“가자.” 하고 별이가 말했어요."
    )


def test_converts_single_curly_pair_and_trailing_comma() -> None:
    assert (
        normalize_dialogue_quotes("‘가자,’ 하고 별이가 속삭였어요.")
        == "“가자.” 하고 별이가 속삭였어요."
    )


def test_keeps_sentences_without_dialogue_unchanged() -> None:
    assert normalize_dialogue_quotes("따옴표 없는 문장이에요.") == "따옴표 없는 문장이에요."


def test_keeps_unbalanced_straight_quote_unchanged() -> None:
    assert normalize_dialogue_quotes('홀수 " 인용부는 그대로 둬요.') == '홀수 " 인용부는 그대로 둬요.'


def test_normalized_dialogue_passes_contract_check() -> None:
    sentences = (
        "별이는 숲으로 걸어가요.",
        normalize_dialogue_quotes('"같이 가자!" 하고 별이가 외쳐요.'),
        "둘은 함께 길을 나서요.",
    )
    assert has_exact_spoken_dialogue(sentences, ("별이",))


def test_standalone_quote_attributed_by_following_sentence() -> None:
    # 실제 Gemini 실패 사례: 대사가 단독 문장, 화자·인용 동사는 다음 문장.
    sentences = (
        "개미가 굴 입구로 나오자 배짱이가 길을 막아 서요.",
        "“왜 날씨 좋은 날에 이렇게 힘들게 일만 하는 거야?”",
        "배짱이가 물어보자 개미는 정성껏 대답을 준비해요.",
    )
    assert has_exact_spoken_dialogue(sentences, ("배짱이", "개미"))


def test_standalone_quote_attributed_by_previous_sentence() -> None:
    sentences = (
        "별이가 웃으며 말해요.",
        "“오늘은 정말 좋은 날이야!”",
        "둘은 함께 길을 나서요.",
    )
    assert has_exact_spoken_dialogue(sentences, ("별이",))


def test_standalone_quote_without_any_attribution_fails() -> None:
    sentences = (
        "숲은 아주 조용했어요.",
        "“오늘은 정말 좋은 날이야!”",
        "바람이 살랑살랑 불어요.",
    )
    assert not has_exact_spoken_dialogue(sentences, ("별이",))


def test_reporting_verb_accepts_connective_forms() -> None:
    sentences = (
        "“오늘은 어디로 갈까?” 별이가 물어보자 달이가 웃으며 대답해요.",
    )
    assert has_exact_spoken_dialogue(sentences, ("별이", "달이"))
