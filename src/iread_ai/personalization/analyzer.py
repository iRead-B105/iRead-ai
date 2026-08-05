from __future__ import annotations

import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from iread_ai.personalization.hangul import (
    COMPLEX_CODA_PARTS,
    count_surface_features,
    decompose_text,
    mask_protected_terms,
    written_syllable_count,
)

TENSE_MAP = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
ASPIRATION_MAP = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
CODA_NEUTRALIZATION_MAP = {
    "ㄲ": "ㄱ",
    "ㅋ": "ㄱ",
    "ㅅ": "ㄷ",
    "ㅆ": "ㄷ",
    "ㅈ": "ㄷ",
    "ㅊ": "ㄷ",
    "ㅌ": "ㄷ",
    "ㅎ": "ㄷ",
    "ㅍ": "ㅂ",
}
NASALIZATION_MAP = {
    "ㄱ": "ㅇ",
    "ㄲ": "ㅇ",
    "ㅋ": "ㅇ",
    "ㄷ": "ㄴ",
    "ㅅ": "ㄴ",
    "ㅆ": "ㄴ",
    "ㅈ": "ㄴ",
    "ㅊ": "ㄴ",
    "ㅌ": "ㄴ",
    "ㅎ": "ㄴ",
    "ㅂ": "ㅁ",
    "ㅍ": "ㅁ",
}
N_INSERTION_NUCLEI = frozenset({"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅖ"})
GLIDE_REDUCTION_MAP = {"ㅑ": "ㅏ", "ㅕ": "ㅓ", "ㅛ": "ㅗ", "ㅠ": "ㅜ"}
DIRECT_DIALOGUE_PATTERNS = (
    re.compile(r'"[^"\n]+"'),
    re.compile(r"“[^”\n]+”"),
    re.compile(r"‘[^’\n]+’"),
)


class AnalysisStatus(StrEnum):
    FULL = "FULL"
    SURFACE_ONLY = "SURFACE_ONLY"
    UNRELIABLE = "UNRELIABLE"


@dataclass(frozen=True, slots=True)
class CandidateAnalysis:
    status: AnalysisStatus
    surface_feature_counts: dict[str, int]
    controllable_surface_feature_counts: dict[str, int]
    protected_surface_feature_counts: dict[str, int]
    phonological_rule_counts: dict[str, int]
    written_syllables: int
    dialogue_sentence_count: int
    pronunciations: tuple[str, ...]
    kiwi_token_count: int
    g2p_review_sentence_count: int
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "surface_feature_counts": dict(self.surface_feature_counts),
            "controllable_surface_feature_counts": dict(self.controllable_surface_feature_counts),
            "protected_surface_feature_counts": dict(self.protected_surface_feature_counts),
            "phonological_rule_counts": dict(self.phonological_rule_counts),
            "written_syllables": self.written_syllables,
            "dialogue_sentence_count": self.dialogue_sentence_count,
            "pronunciations": list(self.pronunciations),
            "kiwi_token_count": self.kiwi_token_count,
            "g2p_review_sentence_count": self.g2p_review_sentence_count,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
        }


def _dialogue_sentence_count(sentences: Iterable[str]) -> int:
    return sum(
        any(pattern.search(sentence) for pattern in DIRECT_DIALOGUE_PATTERNS)
        for sentence in sentences
    )


def _changed_slots(written: tuple[Any, ...], pronounced: tuple[Any, ...]) -> set[str]:
    changed: set[str] = set()
    for index, (written_syllable, pronounced_syllable) in enumerate(
        zip(written, pronounced, strict=True)
    ):
        if written_syllable.onset != pronounced_syllable.onset:
            changed.add(f"onset:{index}")
        if written_syllable.nucleus != pronounced_syllable.nucleus:
            changed.add(f"nucleus:{index}")
        if written_syllable.coda != pronounced_syllable.coda:
            changed.add(f"coda:{index}")
    return changed


def _detect_phonological_rules(written_text: str, pronunciation: str) -> tuple[Counter[str], bool]:
    written = decompose_text(written_text)
    pronounced = decompose_text(pronunciation)
    if len(written) != len(pronounced):
        return Counter(), False

    counts: Counter[str] = Counter()
    explained: set[str] = set()
    changed = _changed_slots(written, pronounced)

    for index, (written_syllable, pronounced_syllable) in enumerate(
        zip(written, pronounced, strict=True)
    ):
        coda_slot = f"coda:{index}"
        onset_slot = f"onset:{index}"
        nucleus_slot = f"nucleus:{index}"
        expected_coda = CODA_NEUTRALIZATION_MAP.get(written_syllable.coda)
        if expected_coda and pronounced_syllable.coda == expected_coda:
            counts["PHONO_CODA_NEUTRALIZATION"] += 1
            explained.add(coda_slot)
        if (
            written_syllable.coda in COMPLEX_CODA_PARTS
            and written_syllable.coda != pronounced_syllable.coda
            and pronounced_syllable.coda
            in {
                "",
                *COMPLEX_CODA_PARTS[written_syllable.coda],
                "ㄱ",
                "ㄴ",
                "ㄹ",
                "ㅁ",
                "ㅂ",
            }
        ):
            counts["PHONO_CLUSTER_SIMPLIFICATION"] += 1
            explained.add(coda_slot)
        expected_nucleus = GLIDE_REDUCTION_MAP.get(written_syllable.nucleus)
        if (
            expected_nucleus
            and pronounced_syllable.nucleus == expected_nucleus
            and pronounced_syllable.onset
            in {written_syllable.onset, TENSE_MAP.get(written_syllable.onset)}
        ):
            counts["PHONO_GLIDE_REDUCTION"] += 1
            explained.add(nucleus_slot)
            if pronounced_syllable.onset != written_syllable.onset:
                explained.add(onset_slot)

    for index in range(len(written) - 1):
        left_w, right_w = written[index], written[index + 1]
        left_p, right_p = pronounced[index], pronounced[index + 1]
        left_coda_slot = f"coda:{index}"
        right_onset_slot = f"onset:{index + 1}"

        palatalized = (
            right_w.onset == "ㅇ"
            and right_w.nucleus == "ㅣ"
            and right_p.onset in {"ㅈ", "ㅊ"}
            and left_w.coda in {"ㄷ", "ㅌ", "ㄾ"}
        )
        if palatalized:
            counts["PHONO_PALATALIZATION"] += 1
            explained.update({left_coda_slot, right_onset_slot})

        if right_w.onset == "ㅇ" and left_w.coda and not palatalized:
            parts = left_w.coda_parts
            moved = right_p.onset
            simple_liaison = (
                len(parts) == 1
                and not left_p.coda
                and moved
                in {
                    parts[0],
                    CODA_NEUTRALIZATION_MAP.get(parts[0]),
                }
            )
            complex_liaison = (
                len(parts) == 2
                and left_p.coda == parts[0]
                and moved in {parts[1], TENSE_MAP.get(parts[1])}
            )
            h_cluster_liaison = (
                len(parts) == 2 and parts[1] == "ㅎ" and not left_p.coda and moved == parts[0]
            )
            if simple_liaison or complex_liaison or h_cluster_liaison:
                counts["PHONO_LIAISON"] += 1
                explained.update({left_coda_slot, right_onset_slot})
                if h_cluster_liaison:
                    counts["PHONO_H_DELETION"] += 1
                if (complex_liaison or h_cluster_liaison) and counts[
                    "PHONO_CLUSTER_SIMPLIFICATION"
                ]:
                    counts["PHONO_CLUSTER_SIMPLIFICATION"] -= 1
                    if not counts["PHONO_CLUSTER_SIMPLIFICATION"]:
                        del counts["PHONO_CLUSTER_SIMPLIFICATION"]

        if (
            left_w.coda == "ㅎ"
            and right_w.onset == "ㅇ"
            and not left_p.coda
            and right_p.onset == "ㅇ"
        ):
            counts["PHONO_H_DELETION"] += 1
            explained.add(left_coda_slot)

        left_base = left_w.coda_parts[-1] if left_w.coda_parts else ""
        aspirated_from_h = (
            right_w.onset == "ㅎ"
            and left_base in ASPIRATION_MAP
            and right_p.onset == ASPIRATION_MAP[left_base]
        )
        aspirated_by_h = (
            "ㅎ" in left_w.coda_parts
            and right_w.onset in ASPIRATION_MAP
            and right_p.onset == ASPIRATION_MAP[right_w.onset]
        )
        if aspirated_from_h or aspirated_by_h:
            counts["PHONO_ASPIRATION"] += 1
            explained.update({left_coda_slot, right_onset_slot})

        if (
            left_w.coda in NASALIZATION_MAP
            and left_p.coda == NASALIZATION_MAP[left_w.coda]
            and right_w.onset in {"ㄴ", "ㅁ"}
            and right_p.onset == right_w.onset
        ):
            counts["PHONO_NASALIZATION"] += 1
            explained.add(left_coda_slot)

        if (
            (left_w.coda, right_w.onset) in {("ㄴ", "ㄹ"), ("ㄹ", "ㄴ")}
            and left_p.coda == "ㄹ"
            and right_p.onset == "ㄹ"
        ):
            counts["PHONO_LIQUIDIZATION"] += 1
            explained.update({left_coda_slot, right_onset_slot})

        if right_w.onset in TENSE_MAP and right_p.onset == TENSE_MAP[right_w.onset]:
            counts["PHONO_TENSIFICATION"] += 1
            explained.add(right_onset_slot)

        if (
            right_w.onset == "ㅇ"
            and right_w.nucleus in N_INSERTION_NUCLEI
            and right_p.onset == "ㄴ"
        ):
            counts["PHONO_N_INSERTION"] += 1
            explained.add(right_onset_slot)

    return counts, not bool(changed - explained)


class KoreanReadingAnalyzer:
    def __init__(
        self,
        *,
        kiwi_factory: Callable[[], Any] | None = None,
        g2p_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._kiwi_factory = kiwi_factory
        self._g2p_factory = g2p_factory
        self._kiwi: Any | None = None
        self._g2p: Any | None = None
        self._kiwi_init_lock = threading.Lock()
        self._g2p_init_lock = threading.Lock()
        self._kiwi_call_lock = threading.Lock()
        self._g2p_call_lock = threading.Lock()

    def _load_kiwi(self) -> Any:
        if self._kiwi is not None:
            return self._kiwi
        with self._kiwi_init_lock:
            if self._kiwi is None:
                if self._kiwi_factory is not None:
                    kiwi = self._kiwi_factory()
                else:
                    from kiwipiepy import Kiwi

                    kiwi = Kiwi(num_workers=0)
                with self._kiwi_call_lock:
                    kiwi.tokenize("", compatible_jamo=True)
                self._kiwi = kiwi
        return self._kiwi

    def _load_g2p(self) -> Any:
        if self._g2p is not None:
            return self._g2p
        with self._g2p_init_lock:
            if self._g2p is None:
                if self._g2p_factory is not None:
                    converter = self._g2p_factory()
                else:
                    from g2pkiwi import G2p

                    converter = G2p()
                with self._g2p_call_lock:
                    converter(
                        "가",
                        descriptive=False,
                        group_vowels=False,
                        to_syl=True,
                    )
                self._g2p = converter
        return self._g2p

    def warmup(self) -> dict[str, object]:
        started = time.perf_counter()
        errors: list[str] = []
        try:
            self._load_kiwi()
        except Exception as exc:
            errors.append(f"Kiwi: {type(exc).__name__}: {exc}")
        try:
            self._load_g2p()
        except Exception as exc:
            errors.append(f"G2P: {type(exc).__name__}: {exc}")
        status = AnalysisStatus.FULL
        if any(error.startswith("G2P:") for error in errors):
            status = AnalysisStatus.SURFACE_ONLY
        elif errors:
            status = AnalysisStatus.UNRELIABLE
        return {
            "status": status.value,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "errors": errors,
        }

    def _token_count(self, sentences: tuple[str, ...]) -> int:
        kiwi = self._load_kiwi()
        count = 0
        with self._kiwi_call_lock:
            for sentence in sentences:
                count += len(kiwi.tokenize(sentence, compatible_jamo=True))
        return count

    def _pronounce(self, sentence: str) -> str:
        converter = self._load_g2p()
        with self._g2p_call_lock:
            return str(
                converter(
                    sentence,
                    descriptive=False,
                    group_vowels=False,
                    to_syl=True,
                )
            )

    def analyze(
        self,
        sentences: tuple[str, ...],
        protected_terms: tuple[str, ...] = (),
    ) -> CandidateAnalysis:
        started = time.perf_counter()
        sentence_list = tuple(str(sentence).strip() for sentence in sentences)
        joined = " ".join(sentence_list)
        surface_counts = count_surface_features(joined)
        controllable_counts = count_surface_features(mask_protected_terms(joined, protected_terms))
        protected_counts = {
            code: count - controllable_counts.get(code, 0)
            for code, count in surface_counts.items()
            if count - controllable_counts.get(code, 0) > 0
        }

        errors: list[str] = []
        kiwi_token_count = 0
        kiwi_failed = False
        try:
            kiwi_token_count = self._token_count(sentence_list)
        except Exception as exc:
            kiwi_failed = True
            errors.append(f"Kiwi: {type(exc).__name__}: {exc}")

        pronunciations: list[str] = []
        rule_counts: Counter[str] = Counter()
        review_count = 0
        g2p_failed = False
        try:
            for sentence in sentence_list:
                pronunciation = self._pronounce(sentence)
                pronunciations.append(pronunciation)
                sentence_rules, reliable = _detect_phonological_rules(
                    sentence,
                    pronunciation,
                )
                rule_counts.update(sentence_rules)
                if not reliable:
                    review_count += 1
        except Exception as exc:
            g2p_failed = True
            rule_counts.clear()
            errors.append(f"G2P: {type(exc).__name__}: {exc}")

        if g2p_failed:
            status = AnalysisStatus.SURFACE_ONLY
        elif kiwi_failed or review_count:
            status = AnalysisStatus.UNRELIABLE
        else:
            status = AnalysisStatus.FULL

        return CandidateAnalysis(
            status=status,
            surface_feature_counts=surface_counts,
            controllable_surface_feature_counts=controllable_counts,
            protected_surface_feature_counts=protected_counts,
            phonological_rule_counts=dict(sorted(rule_counts.items())),
            written_syllables=written_syllable_count(joined),
            dialogue_sentence_count=_dialogue_sentence_count(sentence_list),
            pronunciations=tuple(pronunciations),
            kiwi_token_count=kiwi_token_count,
            g2p_review_sentence_count=review_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            error="; ".join(errors) or None,
        )


__all__ = [
    "AnalysisStatus",
    "CandidateAnalysis",
    "KoreanReadingAnalyzer",
]
