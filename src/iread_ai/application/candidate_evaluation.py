from __future__ import annotations

import re

from iread_ai.personalization.analyzer import (
    AnalysisStatus,
    CandidateAnalysis,
    KoreanReadingAnalyzer,
)
from iread_ai.personalization.generator import PageCandidate
from iread_ai.personalization.hangul import (
    count_surface_features,
    mask_protected_terms,
    written_syllable_count,
)
from iread_ai.personalization.selector import (
    CandidateEvaluation,
    GenerationProfile,
    evaluate_candidate,
)

_DIALOGUE_PATTERN = re.compile(r'("[^"\n]+"|“[^”\n]+”|‘[^’\n]+’)')


def evaluate_page_candidate_resilient(
    candidate: PageCandidate,
    profile: GenerationProfile,
    analyzer: KoreanReadingAnalyzer,
) -> CandidateEvaluation:
    try:
        return evaluate_candidate(
            candidate.candidate_id,
            candidate.sentences,
            profile,
            analyzer,
        )
    except Exception:
        fallback = _surface_only_analysis(
            candidate.sentences,
            profile.protected_terms,
        )
        return evaluate_candidate(
            candidate.candidate_id,
            candidate.sentences,
            profile,
            _StaticAnalyzer(fallback),
        )


class _StaticAnalyzer:
    def __init__(self, result: CandidateAnalysis) -> None:
        self._result = result

    def analyze(
        self,
        _sentences: tuple[str, ...],
        protected_terms: tuple[str, ...] = (),
    ) -> CandidateAnalysis:
        return self._result


def _surface_only_analysis(
    sentences: tuple[str, ...],
    protected_terms: tuple[str, ...],
) -> CandidateAnalysis:
    joined = " ".join(sentences)
    surface = count_surface_features(joined)
    controllable = count_surface_features(
        mask_protected_terms(joined, protected_terms)
    )
    protected = {
        code: count - controllable.get(code, 0)
        for code, count in surface.items()
        if count - controllable.get(code, 0) > 0
    }
    return CandidateAnalysis(
        status=AnalysisStatus.SURFACE_ONLY,
        surface_feature_counts=surface,
        controllable_surface_feature_counts=controllable,
        protected_surface_feature_counts=protected,
        phonological_rule_counts={},
        written_syllables=written_syllable_count(joined),
        dialogue_sentence_count=sum(
            _DIALOGUE_PATTERN.search(sentence) is not None
            for sentence in sentences
        ),
        pronunciations=(),
        kiwi_token_count=0,
        g2p_review_sentence_count=len(sentences),
        latency_ms=0.0,
        error="local linguistic analysis unavailable",
    )


__all__ = ["evaluate_page_candidate_resilient"]
