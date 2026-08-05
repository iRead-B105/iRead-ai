from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from iread_ai.personalization.hangul import written_syllable_count

_DIALOGUE_PATTERN = re.compile(r'("[^"\n]+"|“[^”\n]+”|‘[^’\n]+’)')


class PagePartitionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DynamicStoryPage:
    page_number: int
    start_sentence_index: int
    end_sentence_index: int
    sentences: tuple[str, ...]
    written_syllable_count: int
    direct_dialogue_count: int
    contract_pass: bool
    contract_failures: tuple[str, ...]
    accepted_length_distance: int
    preferred_length_distance: int


@dataclass(frozen=True, slots=True)
class ChapterPartition:
    pages: tuple[DynamicStoryPage, ...]
    contract_pass: bool
    contract_failure_count: int
    contract_penalty: int
    preferred_length_distance: int
    forced_first_break: int | None

    @property
    def page_count(self) -> int:
        return len(self.pages)


def partition_chapter_sentences(
    sentences: Iterable[str],
    *,
    min_pages: int = 2,
    max_pages: int = 4,
    min_sentences_per_page: int = 3,
    max_sentences_per_page: int = 4,
    preferred_min_syllables: int = 50,
    preferred_max_syllables: int = 70,
    accepted_min_syllables: int = 45,
    accepted_max_syllables: int = 80,
    direct_dialogue_per_page: int = 1,
    forced_first_break: int | None = None,
) -> ChapterPartition:
    normalized = tuple(str(sentence).strip() for sentence in sentences)
    _validate_options(
        normalized,
        min_pages=min_pages,
        max_pages=max_pages,
        min_sentences_per_page=min_sentences_per_page,
        max_sentences_per_page=max_sentences_per_page,
        preferred_min_syllables=preferred_min_syllables,
        preferred_max_syllables=preferred_max_syllables,
        accepted_min_syllables=accepted_min_syllables,
        accepted_max_syllables=accepted_max_syllables,
        direct_dialogue_per_page=direct_dialogue_per_page,
        forced_first_break=forced_first_break,
    )

    partitions: list[ChapterPartition] = []
    sentence_count = len(normalized)
    for page_count in range(min_pages, max_pages + 1):
        for cuts in combinations(range(1, sentence_count), page_count - 1):
            boundaries = (0, *cuts, sentence_count)
            sizes = tuple(boundaries[index + 1] - boundaries[index] for index in range(page_count))
            if any(
                size < min_sentences_per_page or size > max_sentences_per_page for size in sizes
            ):
                continue
            if forced_first_break is not None and boundaries[1] != forced_first_break:
                continue
            partitions.append(
                _build_partition(
                    normalized,
                    boundaries,
                    accepted_min_syllables=accepted_min_syllables,
                    accepted_max_syllables=accepted_max_syllables,
                    preferred_min_syllables=preferred_min_syllables,
                    preferred_max_syllables=preferred_max_syllables,
                    direct_dialogue_per_page=direct_dialogue_per_page,
                    forced_first_break=forced_first_break,
                )
            )

    if not partitions:
        suffix = (
            f" with forced_first_break={forced_first_break}"
            if forced_first_break is not None
            else ""
        )
        raise PagePartitionError(
            "sentences cannot be divided into the requested page layout" + suffix
        )

    return min(partitions, key=_partition_rank)


def split_chapter_into_pages(
    sentences: Iterable[str],
    **options: int | None,
) -> ChapterPartition:
    return partition_chapter_sentences(sentences, **options)


def _build_partition(
    sentences: tuple[str, ...],
    boundaries: tuple[int, ...],
    *,
    accepted_min_syllables: int,
    accepted_max_syllables: int,
    preferred_min_syllables: int,
    preferred_max_syllables: int,
    direct_dialogue_per_page: int,
    forced_first_break: int | None,
) -> ChapterPartition:
    pages: list[DynamicStoryPage] = []
    for page_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True),
        start=1,
    ):
        page_sentences = sentences[start:end]
        syllables = written_syllable_count(" ".join(page_sentences))
        dialogue_count = sum(
            _DIALOGUE_PATTERN.search(sentence) is not None for sentence in page_sentences
        )
        accepted_distance = _range_distance(
            syllables,
            accepted_min_syllables,
            accepted_max_syllables,
        )
        preferred_distance = _range_distance(
            syllables,
            preferred_min_syllables,
            preferred_max_syllables,
        )
        failures: list[str] = []
        if accepted_distance:
            failures.append("WRITTEN_SYLLABLE_RANGE")
        if dialogue_count > direct_dialogue_per_page:
            failures.append("DIRECT_DIALOGUE_COUNT")
        pages.append(
            DynamicStoryPage(
                page_number=page_index,
                start_sentence_index=start + 1,
                end_sentence_index=end,
                sentences=page_sentences,
                written_syllable_count=syllables,
                direct_dialogue_count=dialogue_count,
                contract_pass=not failures,
                contract_failures=tuple(failures),
                accepted_length_distance=accepted_distance,
                preferred_length_distance=preferred_distance,
            )
        )

    failure_count = sum(len(page.contract_failures) for page in pages)
    accepted_penalty = sum(page.accepted_length_distance for page in pages)
    dialogue_penalty = sum(
        max(0, page.direct_dialogue_count - direct_dialogue_per_page) for page in pages
    )
    return ChapterPartition(
        pages=tuple(pages),
        contract_pass=failure_count == 0,
        contract_failure_count=failure_count,
        contract_penalty=accepted_penalty + 10 * dialogue_penalty,
        preferred_length_distance=sum(page.preferred_length_distance for page in pages),
        forced_first_break=forced_first_break,
    )


def _partition_rank(partition: ChapterPartition) -> tuple[object, ...]:
    syllable_counts = tuple(page.written_syllable_count for page in partition.pages)
    sentence_shape_penalty = sum(4 - len(page.sentences) for page in partition.pages)
    spread = max(syllable_counts) - min(syllable_counts)
    boundaries = tuple(page.end_sentence_index for page in partition.pages[:-1])
    return (
        partition.contract_failure_count,
        partition.contract_penalty,
        partition.preferred_length_distance,
        sentence_shape_penalty,
        partition.page_count,
        spread,
        boundaries,
    )


def _range_distance(value: int, minimum: int, maximum: int) -> int:
    if value < minimum:
        return minimum - value
    if value > maximum:
        return value - maximum
    return 0


def _validate_options(
    sentences: tuple[str, ...],
    *,
    min_pages: int,
    max_pages: int,
    min_sentences_per_page: int,
    max_sentences_per_page: int,
    preferred_min_syllables: int,
    preferred_max_syllables: int,
    accepted_min_syllables: int,
    accepted_max_syllables: int,
    direct_dialogue_per_page: int,
    forced_first_break: int | None,
) -> None:
    if not sentences:
        raise PagePartitionError("at least one sentence is required")
    if any(not sentence for sentence in sentences):
        raise PagePartitionError("sentences must not contain blank values")
    if min_pages < 1 or min_pages > max_pages:
        raise PagePartitionError("page range is invalid")
    if min_sentences_per_page < 1 or (min_sentences_per_page > max_sentences_per_page):
        raise PagePartitionError("sentence range is invalid")
    if not (
        0
        <= accepted_min_syllables
        <= preferred_min_syllables
        <= preferred_max_syllables
        <= accepted_max_syllables
    ):
        raise PagePartitionError("syllable ranges are inconsistent")
    if direct_dialogue_per_page < 0:
        raise PagePartitionError("direct_dialogue_per_page must be zero or greater")
    minimum_total = min_pages * min_sentences_per_page
    maximum_total = max_pages * max_sentences_per_page
    if not minimum_total <= len(sentences) <= maximum_total:
        raise PagePartitionError(
            f"sentence count must be between {minimum_total} and {maximum_total}"
        )
    if forced_first_break is not None and not (
        min_sentences_per_page <= forced_first_break <= max_sentences_per_page
    ):
        raise PagePartitionError("forced_first_break must fit the first-page sentence range")


__all__ = [
    "ChapterPartition",
    "DynamicStoryPage",
    "PagePartitionError",
    "partition_chapter_sentences",
    "split_chapter_into_pages",
]
