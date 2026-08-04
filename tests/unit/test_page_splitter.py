from __future__ import annotations

import pytest

from iread_ai.personalization.page_splitter import (
    PagePartitionError,
    partition_chapter_sentences,
    split_chapter_into_pages,
)


def _sentence(*, dialogue: bool = False, syllables: int = 15) -> str:
    if dialogue:
        body = "가" * max(1, syllables - 8)
        return f"토끼가 “{body}”라고 말해요."
    return "가" * syllables + "."


def _chapter(
    sentence_count: int,
    dialogue_positions: set[int],
    *,
    syllables: int = 15,
) -> tuple[str, ...]:
    return tuple(
        _sentence(
            dialogue=index in dialogue_positions,
            syllables=syllables,
        )
        for index in range(1, sentence_count + 1)
    )


def test_eight_sentences_are_divided_into_two_valid_pages() -> None:
    result = partition_chapter_sentences(
        _chapter(8, {2, 6}),
    )

    assert result.contract_pass
    assert result.page_count == 2
    assert [len(page.sentences) for page in result.pages] == [4, 4]
    assert all(50 <= page.written_syllable_count <= 70 for page in result.pages)
    assert all(page.direct_dialogue_count == 1 for page in result.pages)
    assert [page.page_number for page in result.pages] == [1, 2]
    assert [(page.start_sentence_index, page.end_sentence_index) for page in result.pages] == [
        (1, 4),
        (5, 8),
    ]


def test_length_selects_three_and_four_pages_without_fixed_page_count() -> None:
    three_pages = partition_chapter_sentences(
        _chapter(12, {2, 6, 10}),
    )
    four_pages = partition_chapter_sentences(
        _chapter(16, {2, 6, 10, 14}),
    )

    assert three_pages.page_count == 3
    assert [len(page.sentences) for page in three_pages.pages] == [4, 4, 4]
    assert four_pages.page_count == 4
    assert [len(page.sentences) for page in four_pages.pages] == [4, 4, 4, 4]


def test_nine_sentences_never_create_a_five_sentence_page() -> None:
    sentences = _chapter(9, {2, 5, 8})

    result = partition_chapter_sentences(sentences)

    assert result.contract_pass
    assert result.page_count == 3
    assert [len(page.sentences) for page in result.pages] == [3, 3, 3]


def test_pages_without_dialogue_are_valid_when_one_is_the_maximum() -> None:
    result = partition_chapter_sentences(_chapter(9, set()))

    assert result.contract_pass
    assert all(page.direct_dialogue_count == 0 for page in result.pages)


def test_forced_first_break_keeps_child_detour_on_first_page() -> None:
    sentences = _chapter(10, {2, 6, 9})

    result = partition_chapter_sentences(
        sentences,
        forced_first_break=4,
    )

    assert result.forced_first_break == 4
    assert result.pages[0].end_sentence_index == 4
    assert result.pages[1].start_sentence_index == 5


def test_best_effort_partition_is_returned_when_syllables_are_too_short() -> None:
    result = partition_chapter_sentences(
        _chapter(6, {2, 5}, syllables=5),
    )

    assert result.page_count == 2
    assert result.contract_pass is False
    assert result.contract_failure_count == 2
    assert all(page.contract_failures == ("WRITTEN_SYLLABLE_RANGE",) for page in result.pages)


def test_split_alias_uses_the_same_deterministic_result() -> None:
    sentences = _chapter(8, {2, 6})

    direct = partition_chapter_sentences(sentences)
    alias = split_chapter_into_pages(sentences)

    assert alias == direct


@pytest.mark.parametrize(
    ("sentences", "forced_first_break", "message"),
    [
        (_chapter(5, {2}), None, "sentence count"),
        (_chapter(8, {2, 6}), 2, "forced_first_break"),
        (_chapter(9, {2, 5, 8}), 4, "cannot be divided"),
    ],
)
def test_invalid_or_impossible_layouts_raise_clear_errors(
    sentences: tuple[str, ...],
    forced_first_break: int | None,
    message: str,
) -> None:
    with pytest.raises(PagePartitionError, match=message):
        partition_chapter_sentences(
            sentences,
            forced_first_break=forced_first_break,
        )


@pytest.mark.parametrize("sentence_count", range(8, 17))
def test_dynamic_layout_never_exceeds_four_sentences(
    sentence_count: int,
) -> None:
    source = _chapter(sentence_count, set())

    result = partition_chapter_sentences(
        source,
        direct_dialogue_per_page=0,
    )

    assert 2 <= result.page_count <= 4
    assert all(3 <= len(page.sentences) <= 4 for page in result.pages)
    assert tuple(sentence for page in result.pages for sentence in page.sentences) == source
