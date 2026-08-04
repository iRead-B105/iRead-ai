from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_SENTENCE_END_PATTERN = re.compile(r"[.!?。！？][”’'\"]?$")
_HANGUL_PATTERN = re.compile(r"[가-힣]")
_WORD_PATTERN = re.compile(r"[가-힣]+")
_SURFACE_PARTICLE_PATTERN = re.compile(
    r"([가-힣]+?)(으로|로|은|는|이|가|을|를|과|와)(?=\s|[,.!?。！？]|$)"
)
_TOPIC_OBJECT_CASE_PATTERN = re.compile(
    r"[가-힣]+[은는][^.!?。！？]{0,30}?"
    r"[가-힣]+[이가]\s+"
    r"(?:먹|마시|읽|보|갖|잡|열|닫|사|쓰)[가-힣]*고\s+싶"
)
_PARTICLE_BY_CODA = {
    "은": True,
    "는": False,
    "이": True,
    "가": False,
    "을": True,
    "를": False,
    "과": True,
    "와": False,
}
_CONTENT_TAG_PREFIXES = ("NN", "VV", "VA", "XR")
_GENERIC_IMAGE_WORDS = frozenset(
    {
        "그림",
        "장면",
        "모습",
        "배경",
        "아이",
        "친구",
        "있",
        "하",
        "되",
    }
)

_kiwi: Any | None = None
_kiwi_failed = False
_kiwi_init_lock = threading.Lock()
_kiwi_call_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class KoreanContentLexeme:
    surface: str
    lookupForms: tuple[str, ...]
    tag: str


def validate_complete_korean_sentence(text: str) -> None:
    sentence = text.strip()
    if not sentence or not _HANGUL_PATTERN.search(sentence):
        raise ValueError("reading text must contain a Korean sentence")
    if "{{" in sentence or "<string>" in sentence:
        raise ValueError("reading text contained an unresolved placeholder")
    if not _SENTENCE_END_PATTERN.search(sentence):
        raise ValueError("reading sentence must end with sentence punctuation")

    words = _WORD_PATTERN.findall(sentence)
    if any(first == second for first, second in zip(words, words[1:], strict=False)):
        raise ValueError("reading sentence repeated the same adjacent word")
    if _TOPIC_OBJECT_CASE_PATTERN.search(sentence):
        raise ValueError("reading sentence used a subject particle for a desired object")

    tokens = _tokenize(sentence)
    if tokens is None:
        return
    _validate_surface_particles(sentence, tokens)
    _validate_token_particles(tokens)
    if not _has_complete_predicate(tokens):
        raise ValueError("reading sentence did not contain a complete predicate ending")
    if _contains_name_suffix_particle_error(tokens):
        raise ValueError("reading sentence contained a malformed name particle")


def validate_image_sentence_answer(
    image_prompt: str,
    choices: Sequence[str],
    answer_index: int,
) -> None:
    if not 0 <= answer_index < len(choices):
        raise ValueError("image sentence answerIndex was outside choices")
    prompt_terms = content_terms(image_prompt)
    overlaps = [len(prompt_terms.intersection(content_terms(choice))) for choice in choices]
    correct = overlaps[answer_index]
    other = max(
        (score for index, score in enumerate(overlaps) if index != answer_index),
        default=-1,
    )
    if correct < 1 or correct <= other:
        raise ValueError("image prompt did not uniquely support the selected sentence")


def content_terms(text: str) -> set[str]:
    tokens = _tokenize(text)
    if tokens is not None:
        return {
            str(token.form)
            for token in tokens
            if str(token.tag).startswith(_CONTENT_TAG_PREFIXES)
            and str(token.form) not in _GENERIC_IMAGE_WORDS
            and len(str(token.form)) >= 1
        }
    return {
        word
        for word in _WORD_PATTERN.findall(text)
        if word not in _GENERIC_IMAGE_WORDS and len(word) >= 2
    }


def content_lexemes(text: str) -> tuple[KoreanContentLexeme, ...] | None:
    tokens = _tokenize(text)
    if tokens is None:
        return None
    result: list[KoreanContentLexeme] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        surface = str(token.form).strip()
        tag = str(token.tag)
        if not surface or not _HANGUL_PATTERN.search(surface):
            continue
        if tag == "NNP":
            continue
        if tag.startswith(("NN", "NP")):
            lookup_forms = (surface,)
        elif tag.startswith(("VV", "VA", "VX", "VCP", "VCN")):
            lookup_forms = (surface, f"{surface}다")
        else:
            continue
        key = (surface, tag)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            KoreanContentLexeme(
                surface=surface,
                lookupForms=lookup_forms,
                tag=tag,
            )
        )
    return tuple(result)


def _validate_token_particles(tokens: Sequence[Any]) -> None:
    for previous, token in zip(tokens, tokens[1:], strict=False):
        particle = str(token.form)
        if not str(token.tag).startswith("J"):
            continue
        if particle not in {*_PARTICLE_BY_CODA, "으로", "로"}:
            continue
        preceding = [character for character in str(previous.form) if "가" <= character <= "힣"]
        if not preceding:
            continue
        coda_index = (ord(preceding[-1]) - 0xAC00) % 28
        if particle in {"으로", "로"}:
            has_non_rieul_coda = coda_index not in {0, 8}
            valid = (particle == "으로") == has_non_rieul_coda
        else:
            valid = _PARTICLE_BY_CODA[particle] == (coda_index != 0)
        if not valid:
            raise ValueError("reading sentence contained incorrect particle agreement")


def _validate_surface_particles(sentence: str, tokens: Sequence[Any]) -> None:
    for match in _SURFACE_PARTICLE_PATTERN.finditer(sentence):
        particle_start = match.end(1)
        particle_end = match.end(2)
        if any(
            str(token.tag).startswith("ETM")
            and int(token.start) < particle_end
            and int(token.start) + int(token.len) > particle_start
            for token in tokens
        ):
            continue
        noun = match.group(1)
        particle = match.group(2)
        coda_index = (ord(noun[-1]) - 0xAC00) % 28
        if particle in {"으로", "로"}:
            has_non_rieul_coda = coda_index not in {0, 8}
            valid = (particle == "으로") == has_non_rieul_coda
        else:
            valid = _PARTICLE_BY_CODA[particle] == (coda_index != 0)
        if not valid:
            raise ValueError("reading sentence contained incorrect particle agreement")


def _contains_name_suffix_particle_error(tokens: Sequence[Any]) -> bool:
    for first, second, third in zip(tokens, tokens[1:], tokens[2:], strict=False):
        if (
            str(first.tag) == "NNP"
            and str(second.tag) == "XSN"
            and str(second.form) == "이"
            and str(third.tag).startswith("J")
        ):
            return True
    return False


def _has_complete_predicate(tokens: Sequence[Any]) -> bool:
    if any(str(token.tag).startswith("EF") for token in tokens):
        return True
    return any(
        str(first.tag).startswith("EC")
        and str(second.tag) == "JX"
        and str(second.form) == "요"
        for first, second in zip(tokens, tokens[1:], strict=False)
    )


def _tokenize(text: str) -> Sequence[Any] | None:
    kiwi = _load_kiwi()
    if kiwi is None:
        return None
    with _kiwi_call_lock:
        return kiwi.tokenize(text, compatible_jamo=True)


def _load_kiwi() -> Any | None:
    global _kiwi, _kiwi_failed
    if _kiwi is not None:
        return _kiwi
    if _kiwi_failed:
        return None
    with _kiwi_init_lock:
        if _kiwi is not None:
            return _kiwi
        if _kiwi_failed:
            return None
        try:
            from kiwipiepy import Kiwi

            kiwi = Kiwi(num_workers=0)
            kiwi.tokenize("", compatible_jamo=True)
            _kiwi = kiwi
        except (ImportError, OSError, RuntimeError):
            _kiwi_failed = True
            return None
    return _kiwi


__all__ = [
    "KoreanContentLexeme",
    "content_lexemes",
    "content_terms",
    "validate_complete_korean_sentence",
    "validate_image_sentence_answer",
]
