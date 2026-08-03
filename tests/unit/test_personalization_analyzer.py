from __future__ import annotations

from dataclasses import dataclass

from iread_ai.personalization.analyzer import AnalysisStatus, KoreanReadingAnalyzer


@dataclass(frozen=True)
class FakeToken:
    form: str


class FakeKiwi:
    def tokenize(
        self,
        text: str,
        *,
        compatible_jamo: bool = True,
    ) -> list[FakeToken]:
        del compatible_jamo
        return [FakeToken(part) for part in text.split()]


class FakeG2p:
    def __call__(self, text: str, **_: object) -> str:
        return {
            "국물": "궁물",
            "같이": "가치",
            "맑은": "말근",
            "괜찮아": "괜차나",
            "않아요": "아나요",
            "많아요": "마나요",
            "싫어요": "시러요",
            "되어": "돼",
        }.get(text, text)


class FailingG2p:
    def __call__(self, text: str, **_: object) -> str:
        if text == "가":
            return text
        raise RuntimeError("converter unavailable")


def test_surface_features_include_jamo_and_structural_codes() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FakeG2p,
    )

    result = analyzer.analyze(('꽉 밖 값 카 야 "좋아!"',))

    assert result.status is AnalysisStatus.FULL
    assert result.surface_feature_counts["ONSET_ㄲ"] == 1
    assert result.surface_feature_counts["NUCLEUS_ㅘ"] == 1
    assert result.surface_feature_counts["CODA_ㄲ"] == 1
    assert result.surface_feature_counts["HAS_BATCHIM"] == 4
    assert result.surface_feature_counts["HAS_COMPLEX_CODA"] == 1
    assert result.surface_feature_counts["HAS_DOUBLE_CODA"] == 1
    assert result.surface_feature_counts["HAS_TENSE_ONSET"] == 1
    assert result.surface_feature_counts["HAS_ASPIRATED_ONSET"] == 1
    assert result.surface_feature_counts["HAS_COMPOUND_VOWEL"] == 1
    assert result.surface_feature_counts["HAS_GLIDE_VOWEL"] == 1
    assert result.dialogue_sentence_count == 1


def test_protected_terms_are_removed_only_from_controllable_surface_counts() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FakeG2p,
    )

    result = analyzer.analyze(("밖에서 밖을 봐요.",), protected_terms=("밖",))

    assert result.surface_feature_counts["CODA_ㄲ"] == 2
    assert "CODA_ㄲ" not in result.controllable_surface_feature_counts
    assert result.protected_surface_feature_counts["CODA_ㄲ"] == 2
    assert result.protected_surface_feature_counts["HAS_DOUBLE_CODA"] == 2
    assert result.written_syllables > 2


def test_g2p_detects_rules_and_unaligned_output_is_not_trusted() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FakeG2p,
    )

    full = analyzer.analyze(("국물", "같이"))
    unreliable = analyzer.analyze(("되어",))

    assert full.status is AnalysisStatus.FULL
    assert full.phonological_rule_counts["PHONO_NASALIZATION"] == 1
    assert full.phonological_rule_counts["PHONO_PALATALIZATION"] == 1
    assert unreliable.status is AnalysisStatus.UNRELIABLE
    assert unreliable.g2p_review_sentence_count == 1


def test_complex_coda_liaison_is_not_double_counted_as_simplification() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FakeG2p,
    )

    result = analyzer.analyze(("맑은",))

    assert result.status is AnalysisStatus.FULL
    assert result.phonological_rule_counts == {"PHONO_LIAISON": 1}


def test_h_complex_coda_before_vowel_is_fully_explained() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FakeG2p,
    )

    result = analyzer.analyze(
        ("괜찮아", "않아요", "많아요", "싫어요"),
    )

    assert result.status is AnalysisStatus.FULL
    assert result.g2p_review_sentence_count == 0
    assert result.phonological_rule_counts == {
        "PHONO_H_DELETION": 4,
        "PHONO_LIAISON": 4,
    }


def test_g2p_failure_returns_surface_only_with_error_not_zero_claim() -> None:
    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=FakeKiwi,
        g2p_factory=FailingG2p,
    )

    result = analyzer.analyze(("국물",))

    assert result.status is AnalysisStatus.SURFACE_ONLY
    assert result.phonological_rule_counts == {}
    assert result.surface_feature_counts["HAS_BATCHIM"] == 2
    assert "G2P" in (result.error or "")


def test_kiwi_and_g2p_are_lazy_cached_instances() -> None:
    calls = {"kiwi": 0, "g2p": 0}

    def kiwi_factory() -> FakeKiwi:
        calls["kiwi"] += 1
        return FakeKiwi()

    def g2p_factory() -> FakeG2p:
        calls["g2p"] += 1
        return FakeG2p()

    analyzer = KoreanReadingAnalyzer(
        kiwi_factory=kiwi_factory,
        g2p_factory=g2p_factory,
    )
    assert calls == {"kiwi": 0, "g2p": 0}

    analyzer.warmup()
    analyzer.analyze(("가요.",))
    analyzer.analyze(("와요.",))

    assert calls == {"kiwi": 1, "g2p": 1}
