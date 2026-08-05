from __future__ import annotations

from dataclasses import replace

from iread_ai.personalization.analyzer import AnalysisStatus, CandidateAnalysis
from iread_ai.personalization.selector import (
    CandidateEvaluation,
    ContentContract,
    GenerationProfile,
    SkillPolicy,
    evaluate_candidate,
    select_best,
)


def _analysis(
    *,
    status: AnalysisStatus = AnalysisStatus.FULL,
    syllables: int = 60,
    dialogue: int = 1,
    surface: dict[str, int] | None = None,
    phonology: dict[str, int] | None = None,
) -> CandidateAnalysis:
    counts = surface or {}
    return CandidateAnalysis(
        status=status,
        surface_feature_counts=dict(counts),
        controllable_surface_feature_counts=dict(counts),
        protected_surface_feature_counts={},
        phonological_rule_counts=dict(phonology or {}),
        written_syllables=syllables,
        dialogue_sentence_count=dialogue,
        pronunciations=(),
        kiwi_token_count=0,
        g2p_review_sentence_count=0,
        latency_ms=1.0,
    )


class StubAnalyzer:
    def __init__(self, analysis: CandidateAnalysis) -> None:
        self.analysis = analysis
        self.protected_terms: tuple[str, ...] | None = None

    def analyze(
        self,
        sentences: tuple[str, ...],
        protected_terms: tuple[str, ...] = (),
    ) -> CandidateAnalysis:
        del sentences
        self.protected_terms = protected_terms
        return self.analysis


def _evaluation(
    candidate_id: str,
    *,
    contract_pass: bool = True,
    excluded: int = 0,
    limited: int = 0,
    status: AnalysisStatus = AnalysisStatus.FULL,
    risk: float = 0.0,
    distance: int = 0,
    target_distance: int = 0,
    contract_penalty: int = 0,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        sentences=("가.",) * 4,
        analysis=_analysis(status=status),
        contract_pass=contract_pass,
        contract_failures=() if contract_pass else ("WRITTEN_SYLLABLE_RANGE",),
        excluded_overage=excluded,
        limited_overage=limited,
        target_distance=target_distance,
        feature_occurrences={},
        feature_risks={},
        total_risk=risk * 6,
        risk_per_10=risk,
        preferred_length_distance=distance,
        contract_penalty=contract_penalty,
    )


def test_evaluate_candidate_applies_contract_and_policy_counts() -> None:
    analyzer = StubAnalyzer(
        _analysis(
            surface={"CODA_ㅅ": 3, "ONSET_ㄲ": 2},
            phonology={"PHONO_LIAISON": 2},
        )
    )
    profile = GenerationProfile(
        skills=(
            SkillPolicy(
                code="CODA_ㅅ",
                role="EXCLUDED",
                max_occurrences=1,
                unit_penalty=1.5,
            ),
            SkillPolicy(
                code="PHONO_LIAISON",
                role="LIMITED",
                max_occurrences=1,
                unit_penalty=2.0,
            ),
            SkillPolicy(
                code="ONSET_ㄲ",
                role="TARGET",
                target_min=1,
                target_max=1,
                unit_penalty=0.5,
            ),
        ),
        protected_terms=("토끼",),
    )

    result = evaluate_candidate(
        "candidate-1",
        ("첫 문장.", "“좋아!”라고 말해요.", "셋째 문장.", "마지막 문장."),
        profile,
        analyzer,  # type: ignore[arg-type]
    )

    assert result.contract_pass
    assert result.excluded_overage == 2
    assert result.limited_overage == 1
    assert result.target_distance == 1
    assert result.feature_occurrences["PHONO_LIAISON"] == 2
    assert result.total_risk == 9.0
    assert analyzer.protected_terms == ("토끼",)


def test_contract_failure_is_detected() -> None:
    analyzer = StubAnalyzer(_analysis(syllables=49, dialogue=0))
    profile = GenerationProfile(content_contract=ContentContract())

    result = evaluate_candidate(
        "short",
        ("하나.", "둘.", "셋."),
        profile,
        analyzer,  # type: ignore[arg-type]
    )

    assert not result.contract_pass
    assert set(result.contract_failures) == {
        "SENTENCE_COUNT",
        "WRITTEN_SYLLABLE_RANGE",
    }


def test_contract_treats_direct_dialogue_as_a_maximum() -> None:
    profile = GenerationProfile(content_contract=ContentContract())

    without_dialogue = evaluate_candidate(
        "without-dialogue",
        ("하나.", "둘.", "셋.", "넷."),
        profile,
        StubAnalyzer(_analysis(dialogue=0)),  # type: ignore[arg-type]
    )
    with_too_much_dialogue = evaluate_candidate(
        "too-much-dialogue",
        ("하나.", "둘.", "셋.", "넷."),
        profile,
        StubAnalyzer(_analysis(dialogue=2)),  # type: ignore[arg-type]
    )

    assert without_dialogue.contract_pass
    assert "DIRECT_DIALOGUE_COUNT" in with_too_much_dialogue.contract_failures


def test_select_best_uses_required_lexicographic_order() -> None:
    rows = [
        _evaluation("contract-fail", contract_pass=False),
        _evaluation("excluded", excluded=1),
        _evaluation("limited", limited=1),
        _evaluation("target-miss", target_distance=1),
        _evaluation("surface", status=AnalysisStatus.SURFACE_ONLY),
        _evaluation("unreliable", status=AnalysisStatus.UNRELIABLE),
        _evaluation("risk-high", risk=2.0),
        _evaluation("length-far", risk=1.0, distance=3),
        _evaluation("b", risk=1.0, distance=0),
        _evaluation("a", risk=1.0, distance=0),
    ]

    assert select_best(rows).candidate_id == "a"

    contract_precedence = [
        _evaluation("pass", excluded=3, risk=10),
        _evaluation("fail", contract_pass=False, risk=0),
    ]
    assert select_best(contract_precedence).candidate_id == "pass"

    all_failed = [
        _evaluation(
            "far",
            contract_pass=False,
            contract_penalty=30,
            risk=0,
        ),
        _evaluation(
            "near",
            contract_pass=False,
            contract_penalty=10,
            risk=5,
        ),
    ]
    assert select_best(all_failed).candidate_id == "near"

    g2p_precedence = [
        _evaluation("full", status=AnalysisStatus.FULL, risk=5),
        _evaluation("surface", status=AnalysisStatus.SURFACE_ONLY, risk=0),
    ]
    assert select_best(g2p_precedence).candidate_id == "full"


def test_select_best_rejects_empty_input() -> None:
    try:
        select_best(())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("select_best must reject empty input")


def test_to_dict_returns_serializable_nested_data() -> None:
    result = _evaluation("candidate")
    changed = replace(result, feature_occurrences={"CODA_ㅅ": 2})

    document = changed.to_dict()

    assert document["candidate_id"] == "candidate"
    assert document["analysis"]["status"] == "FULL"  # type: ignore[index]
    assert document["feature_occurrences"] == {"CODA_ㅅ": 2}
