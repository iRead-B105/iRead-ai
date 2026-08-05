from __future__ import annotations

from collections.abc import Mapping

import pytest

from iread_ai.personalization.analyzer import AnalysisStatus, CandidateAnalysis
from iread_ai.personalization.generator import (
    PageCandidate,
    PageGenerationContext,
    RepairBatch,
    RepairReplacement,
)
from iread_ai.personalization.repair_policy import (
    apply_repair_batch,
    build_repair_plan,
    child_input_signal_preserved,
    evaluate_repair,
    has_exact_spoken_dialogue,
    has_hard_repair_trigger,
)
from iread_ai.personalization.selector import (
    CandidateEvaluation,
    GenerationProfile,
    SkillPolicy,
)


def _analysis(
    *,
    status: AnalysisStatus = AnalysisStatus.FULL,
    surface_counts: Mapping[str, int] | None = None,
    phonological_counts: Mapping[str, int] | None = None,
    written_syllables: int = 60,
    dialogue_sentence_count: int = 1,
) -> CandidateAnalysis:
    return CandidateAnalysis(
        status=status,
        surface_feature_counts=dict(surface_counts or {}),
        controllable_surface_feature_counts=dict(surface_counts or {}),
        protected_surface_feature_counts={},
        phonological_rule_counts=dict(phonological_counts or {}),
        written_syllables=written_syllables,
        dialogue_sentence_count=dialogue_sentence_count,
        pronunciations=(),
        kiwi_token_count=20,
        g2p_review_sentence_count=0,
        latency_ms=1.0,
    )


def _candidate(
    *,
    candidate_id: str = "source",
    sentences: tuple[str, str, str, str] | None = None,
    with_branch: bool = False,
) -> PageCandidate:
    page_sentences = sentences or (
        "거북이는 천천히 언덕을 올라가며 웃었어요.",
        "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
        "거북이는 돌길을 천천히 걸어가요.",
        "토끼는 결승선 옆에서 손을 흔들어요.",
    )
    if with_branch:
        return PageCandidate(
            candidate_id=candidate_id,
            sentences=page_sentences,
            question="거북이는 어느 길로 갈까요?",
            choices=("꽃길로 가요.", "돌길로 가요.", "숲길로 가요."),
        )
    return PageCandidate(
        candidate_id=candidate_id,
        sentences=page_sentences,
    )


def _context(
    *,
    child_input: str = "천천히 가도 괜찮아",
    page_number: int = 1,
) -> PageGenerationContext:
    return PageGenerationContext(
        story_title="토끼와 거북이",
        story_context="토끼와 거북이가 숲속 경주를 하고 있어요.",
        locked_event="거북이가 아이의 응원을 듣고 다시 걸어요.",
        page_number=page_number,
        child_input=child_input,
        characters=("토끼", "거북이"),
    )


def _evaluation(
    candidate: PageCandidate,
    *,
    feature_occurrences: Mapping[str, int] | None = None,
    risk_per_10: float = 1.0,
    failures: tuple[str, ...] = (),
    contract_penalty: int = 0,
    status: AnalysisStatus = AnalysisStatus.FULL,
    written_syllables: int = 60,
    dialogue_sentence_count: int = 1,
) -> CandidateEvaluation:
    occurrences = dict(feature_occurrences or {})
    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        sentences=candidate.sentences,
        analysis=_analysis(
            status=status,
            surface_counts=occurrences,
            phonological_counts={
                code: count for code, count in occurrences.items() if code.startswith("PHONO_")
            },
            written_syllables=written_syllables,
            dialogue_sentence_count=dialogue_sentence_count,
        ),
        contract_pass=not failures,
        contract_failures=failures,
        excluded_overage=0,
        limited_overage=0,
        target_distance=0,
        feature_occurrences=occurrences,
        feature_risks={code: float(count) for code, count in occurrences.items()},
        total_risk=risk_per_10 * 6,
        risk_per_10=risk_per_10,
        preferred_length_distance=0,
        contract_penalty=contract_penalty,
    )


class SentenceFeatureAnalyzer:
    def __init__(self, code: str, marker: str) -> None:
        self.code = code
        self.marker = marker

    def analyze(
        self,
        sentences: tuple[str, ...],
        protected_terms: tuple[str, ...] = (),
    ) -> CandidateAnalysis:
        del protected_terms
        count = sum(self.marker in sentence for sentence in sentences)
        if self.code.startswith("PHONO_"):
            return _analysis(phonological_counts={self.code: count})
        return _analysis(surface_counts={self.code: count})


def _decision(
    *,
    source_candidate: PageCandidate,
    proposal_candidate: PageCandidate,
    profile: GenerationProfile,
    source_occurrences: Mapping[str, int],
    proposal_occurrences: Mapping[str, int],
    changed: tuple[int, ...],
    editable: tuple[int, ...] | None = None,
    context: PageGenerationContext | None = None,
    source_risk: float = 2.0,
    proposal_risk: float = 1.0,
) -> object:
    return evaluate_repair(
        source_candidate=source_candidate,
        source=_evaluation(
            source_candidate,
            feature_occurrences=source_occurrences,
            risk_per_10=source_risk,
        ),
        proposal_candidate=proposal_candidate,
        proposal=_evaluation(
            proposal_candidate,
            feature_occurrences=proposal_occurrences,
            risk_per_10=proposal_risk,
        ),
        context=context or _context(),
        profile=profile,
        editable_indexes=editable or changed,
        changed_sentence_numbers=changed,
    )


def test_limited_overage_is_diagnostic_but_not_a_hard_repair_trigger() -> None:
    source = _candidate()
    skill = SkillPolicy(
        code="CODA_ㄹ",
        role="LIMITED",
        max_occurrences=0,
    )
    profile = GenerationProfile(skills=(skill,))
    plan = build_repair_plan(
        source,
        _evaluation(source, feature_occurrences={"CODA_ㄹ": 1}),
        profile,
        _context(),
        SentenceFeatureAnalyzer("CODA_ㄹ", "돌길"),
    )

    assert has_hard_repair_trigger(plan) is False
    assert plan["trigger_reasons"] == []
    assert plan["violations"][0]["kind"] == "LIMITED_OVERAGE"
    assert plan["violations"][0]["excess"] == 1


def test_excluded_overage_is_a_hard_repair_trigger() -> None:
    source = _candidate()
    skill = SkillPolicy(
        code="CODA_ㄹ",
        role="EXCLUDED",
        max_occurrences=0,
    )
    profile = GenerationProfile(skills=(skill,))
    plan = build_repair_plan(
        source,
        _evaluation(source, feature_occurrences={"CODA_ㄹ": 1}),
        profile,
        _context(),
        SentenceFeatureAnalyzer("CODA_ㄹ", "돌길"),
    )

    assert has_hard_repair_trigger(plan) is True
    assert plan["trigger_reasons"] == ["EXCLUDED_OVERAGE:CODA_ㄹ"]
    assert plan["editable_sentence_indexes"] == [3]


def test_content_contract_failure_is_a_hard_repair_trigger() -> None:
    source = _candidate()
    evaluation = _evaluation(
        source,
        failures=("WRITTEN_SYLLABLE_RANGE",),
        contract_penalty=12,
        written_syllables=87,
    )

    plan = build_repair_plan(
        source,
        evaluation,
        GenerationProfile(),
        _context(),
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is True
    assert "CONTRACT:WRITTEN_SYLLABLE_RANGE" in plan["trigger_reasons"]
    assert 1 <= len(plan["editable_sentence_indexes"]) <= 2


def test_missing_child_input_is_a_hard_trigger_for_first_sentence() -> None:
    source = _candidate(
        sentences=(
            "거북이는 힘찬 응원을 듣고 천천히 걸었어요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            "거북이는 돌길을 천천히 걸어가요.",
            "토끼는 결승선 옆에서 손을 흔들어요.",
        )
    )

    plan = build_repair_plan(
        source,
        _evaluation(source),
        GenerationProfile(),
        _context(child_input="방구소리"),
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is True
    assert plan["trigger_reasons"] == ["STORY:CHILD_INPUT_NOT_REFLECTED"]
    assert plan["editable_sentence_indexes"] == [1, 2]
    assert plan["first_sentence_edit_allowed"] is True
    assert plan["violations"][0]["code"] == "CHILD_INPUT_NOT_REFLECTED"


def test_single_syllable_child_input_can_be_verified() -> None:
    assert child_input_signal_preserved("뿡 소리가 숲에 울려요.", "뿡") is True
    assert child_input_signal_preserved("종소리가 숲에 울려요.", "뿡") is False


def test_missing_child_input_does_not_trigger_after_first_page() -> None:
    source = _candidate(
        sentences=(
            "거북이는 힘찬 응원을 듣고 천천히 걸었어요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            "거북이는 돌길을 천천히 걸어가요.",
            "토끼는 결승선 옆에서 손을 흔들어요.",
        )
    )

    plan = build_repair_plan(
        source,
        _evaluation(source),
        GenerationProfile(),
        _context(child_input="방구소리", page_number=2),
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is False
    assert plan["trigger_reasons"] == []
    assert plan["editable_sentence_indexes"] == []


def test_invalid_dialogue_format_is_a_hard_repair_trigger() -> None:
    sentences = list(_candidate().sentences)
    sentences[1] = '토끼가 거북이에게 "우리 함께 끝까지 가자!"라고 말했어요.'
    source = _candidate(sentences=tuple(sentences))  # type: ignore[arg-type]

    plan = build_repair_plan(
        source,
        _evaluation(source),
        GenerationProfile(),
        _context(),
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is True
    assert "CONTRACT:CURLY_DIALOGUE_FORMAT" in plan["trigger_reasons"]
    assert plan["editable_sentence_indexes"] == [2]


def test_natural_repetition_verb_counts_as_spoken_dialogue() -> None:
    sentences = (
        "토끼는 나무 아래에서 쉬어요.",
        "거북이가 “으쌰으쌰!” 하고 되뇌어요.",
        "풀잎이 박자에 맞춰 흔들려요.",
    )

    assert has_exact_spoken_dialogue(
        sentences,
        ("토끼", "거북이"),
    )


def test_named_speaker_can_finish_dialogue_with_a_natural_reaction() -> None:
    sentences = (
        "거북이는 노란 잎을 펼쳐요.",
        "친구가 “그 잎, 아직 안 날아갔네!” 하고 웃어요.",
        "바람이 잎 가장자리를 흔들어요.",
    )

    assert has_exact_spoken_dialogue(sentences, ("거북이", "친구"))


@pytest.mark.parametrize(
    "meta_sentence",
    (
        "아이의 응원에 거북이가 힘을 내요.",
        "아이가 거북이에게 말해요.",
        "독자가 거북이를 응원해요.",
    ),
)
def test_meta_child_reference_is_a_hard_trigger_for_only_its_sentence(
    meta_sentence: str,
) -> None:
    source = _candidate(
        sentences=(
            "거북이는 천천히 언덕으로 가요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            meta_sentence,
            "거북이는 다시 한 걸음 앞으로 가요.",
        )
    )

    plan = build_repair_plan(
        source,
        _evaluation(source),
        GenerationProfile(),
        _context(),
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is True
    assert plan["trigger_reasons"] == ["STORY:META_CHILD_REFERENCE"]
    assert plan["editable_sentence_indexes"] == [3]
    assert any(
        violation["code"] == "META_CHILD_REFERENCE"
        and violation["affected_sentence_indexes"] == [3]
        for violation in plan["violations"]
    )


def test_child_reference_is_allowed_when_child_is_an_actual_story_character() -> None:
    source = _candidate(
        sentences=(
            "아이와 거북이는 천천히 언덕으로 가요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            "아이의 응원에 거북이가 힘을 내요.",
            "거북이는 다시 한 걸음 앞으로 가요.",
        )
    )
    context = PageGenerationContext(
        story_title="아이와 토끼와 거북이",
        story_context="아이가 토끼와 거북이의 친구로 여행하고 있어요.",
        locked_event="아이가 거북이를 응원해요.",
        child_input="천천히 가도 괜찮아",
        characters=("아이", "토끼", "거북이"),
    )

    plan = build_repair_plan(
        source,
        _evaluation(source),
        GenerationProfile(),
        context,
        SentenceFeatureAnalyzer("CODA_ㄹ", "없는표식"),
    )

    assert has_hard_repair_trigger(plan) is False
    assert "STORY:META_CHILD_REFERENCE" not in plan["trigger_reasons"]
    assert all(violation["code"] != "META_CHILD_REFERENCE" for violation in plan["violations"])


def test_apply_repair_preserves_question_choices_and_changes_only_two_sentences() -> None:
    source = _candidate(with_branch=True)
    batch = RepairBatch(
        source_candidate_id=source.candidate_id,
        repair_status="REPAIRED",
        replacements=(
            RepairReplacement(
                sentence_index=2,
                sentence="토끼가 거북이에게 “천천히 가도 괜찮아!”라고 말했어요.",
            ),
            RepairReplacement(
                sentence_index=4,
                sentence="토끼는 결승선에서 거북이를 기다려요.",
            ),
        ),
        raw_output="{}",
        elapsed_ms=1.0,
    )

    proposal, changed = apply_repair_batch(source, batch)

    assert changed == (2, 4)
    assert proposal.sentences[0] == source.sentences[0]
    assert proposal.sentences[2] == source.sentences[2]
    assert proposal.question == source.question
    assert proposal.choices == source.choices

    with pytest.raises(ValueError, match="at most two"):
        RepairBatch(
            source_candidate_id=source.candidate_id,
            repair_status="REPAIRED",
            replacements=(
                RepairReplacement(sentence_index=1, sentence="첫 문장을 고쳐요."),
                RepairReplacement(sentence_index=2, sentence="둘째 문장을 고쳐요."),
                RepairReplacement(sentence_index=3, sentence="셋째 문장을 고쳐요."),
            ),
            raw_output="{}",
            elapsed_ms=1.0,
        )


def test_low_semantic_overlap_repair_is_rejected() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "거북이는 달빛 속에서 우산을 접어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(3,),
    )

    assert decision.accepted is False
    assert "SEMANTIC_OVERLAP_LOW:3" in decision.reasons


def test_dialogue_speaker_change_is_rejected_even_when_names_are_preserved() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[1] = "거북이가 토끼에게 “우리 함께 끝까지 가자!”라고 말했어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(2,),
    )

    assert decision.accepted is False
    assert decision.source_speaker == "토끼"
    assert decision.proposal_speaker == "거북이"
    assert "DIALOGUE_SPEAKER_CHANGED" in decision.reasons


def test_missing_dialogue_speaker_can_be_filled_with_an_existing_character() -> None:
    source = _candidate(
        sentences=(
            "거북이는 천천히 언덕으로 가요.",
            "“우리 함께 끝까지 가자!”라고 말했어요.",
            "거북이는 돌길을 천천히 걸어가요.",
            "토끼는 결승선 옆에서 손을 흔들어요.",
        )
    )
    proposal_sentences = list(source.sentences)
    proposal_sentences[1] = "토끼가 “우리 함께 끝까지 가자!”라고 말했어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(proposal_sentences),  # type: ignore[arg-type]
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=GenerationProfile(),
        source_occurrences={},
        proposal_occurrences={},
        changed=(2,),
    )

    assert decision.source_speaker is None
    assert decision.proposal_speaker == "토끼"
    assert "CHARACTER_ROLE_CHANGED:2" not in decision.reasons
    assert "DIALOGUE_SPEAKER_CHANGED" not in decision.reasons
    assert decision.accepted is True


def test_character_change_is_rejected() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "토끼는 돌길을 차분히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(3,),
    )

    assert decision.accepted is False
    assert "CHARACTER_ROLE_CHANGED:3" in decision.reasons
    assert any(reason.startswith("PROTECTED_TERM_LOST:") for reason in decision.reasons)


def test_negation_polarity_change_is_rejected() -> None:
    source_sentences = list(_candidate().sentences)
    source_sentences[2] = "거북이는 돌길을 걷지 않아요."
    source = _candidate(sentences=tuple(source_sentences))  # type: ignore[arg-type]
    proposal_sentences = list(source.sentences)
    proposal_sentences[2] = "거북이는 돌길을 천천히 걸어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(proposal_sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(3,),
    )

    assert decision.accepted is False
    assert "NEGATION_CHANGED:3" in decision.reasons


def test_child_input_signal_loss_is_rejected() -> None:
    source_sentences = list(_candidate().sentences)
    source_sentences[0] = "거북이는 아이가 천천히 가도 괜찮다고 응원하자 웃었어요."
    source = _candidate(sentences=tuple(source_sentences))  # type: ignore[arg-type]
    sentences = list(source.sentences)
    sentences[0] = "거북이는 아이가 힘내라고 응원하자 웃었어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(1,),
    )

    assert decision.accepted is False
    assert "CHILD_INPUT_SIGNAL_LOST" in decision.reasons


def test_child_input_signal_must_be_added_when_source_already_omitted_it() -> None:
    source = _candidate(
        sentences=(
            "거북이는 힘찬 응원을 듣고 천천히 걸었어요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            "거북이는 돌길을 천천히 걸어가요.",
            "토끼는 결승선 옆에서 손을 흔들어요.",
        )
    )
    proposal_sentences = list(source.sentences)
    proposal_sentences[2] = "거북이는 돌길을 차분히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(proposal_sentences),  # type: ignore[arg-type]
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=GenerationProfile(),
        source_occurrences={},
        proposal_occurrences={},
        changed=(3,),
        context=_context(child_input="방구소리"),
    )

    assert decision.accepted is False
    assert "CHILD_INPUT_SIGNAL_MISSING" in decision.reasons


def test_adding_missing_child_input_signal_is_an_accepted_improvement() -> None:
    source = _candidate(
        sentences=(
            "거북이는 힘찬 응원을 듣고 천천히 걸었어요.",
            "토끼가 거북이에게 “우리 함께 끝까지 가자!”라고 말했어요.",
            "거북이는 돌길을 천천히 걸어가요.",
            "토끼는 결승선 옆에서 손을 흔들어요.",
        )
    )
    proposal_sentences = list(source.sentences)
    proposal_sentences[0] = "거북이는 방구 소리를 듣고 천천히 걸었어요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(proposal_sentences),  # type: ignore[arg-type]
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=GenerationProfile(),
        source_occurrences={},
        proposal_occurrences={},
        changed=(1,),
        context=_context(child_input="방구소리"),
        source_risk=1.0,
        proposal_risk=1.0,
    )

    assert decision.accepted is True
    assert "CHILD_INPUT_SIGNAL_ADDED" in decision.improvements


def test_per_skill_worsening_is_rejected_despite_better_total_risk() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "거북이는 돌길을 차분히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(
        skills=(
            SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),
            SkillPolicy("CODA_ㄹ", "LIMITED", max_occurrences=1),
        )
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 1, "CODA_ㄹ": 5},
        proposal_occurrences={"CODA_ㅆ": 2, "CODA_ㄹ": 1},
        changed=(3,),
        source_risk=2.0,
        proposal_risk=1.0,
    )

    assert decision.accepted is False
    assert "SKILL_WORSENED:CODA_ㅆ" in decision.reasons
    assert "SKILL_OVERAGE_REDUCED:CODA_ㄹ" in decision.improvements
    assert "NORMALIZED_RISK_REDUCED" in decision.improvements


def test_large_aggregate_limited_overage_triggers_repair() -> None:
    source = _candidate()
    profile = GenerationProfile(
        skills=(
            SkillPolicy(
                "HAS_TENSE_ONSET",
                "LIMITED",
                max_occurrences=1,
            ),
        )
    )
    plan = build_repair_plan(
        source,
        _evaluation(
            source,
            feature_occurrences={"HAS_TENSE_ONSET": 4},
        ),
        profile,
        _context(child_input=""),
        SentenceFeatureAnalyzer("HAS_TENSE_ONSET", "거북"),
    )

    assert plan["limited_overage_total"] == 3
    assert plan["limited_overage_repair_threshold"] == 3
    assert "LIMITED_OVERAGE_TOTAL:3" in plan["trigger_reasons"]
    assert has_hard_repair_trigger(plan) is True
    assert plan["editable_sentence_indexes"]


def test_limited_skill_increase_within_allowance_is_not_skill_worsening() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "거북이는 오솔길을 따라 천천히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(
        skills=(SkillPolicy("HAS_TENSE_ONSET", "LIMITED", max_occurrences=1),)
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"HAS_TENSE_ONSET": 0},
        proposal_occurrences={"HAS_TENSE_ONSET": 1},
        changed=(3,),
        context=_context(child_input=""),
        source_risk=2.0,
        proposal_risk=1.0,
    )

    assert "SKILL_WORSENED:HAS_TENSE_ONSET" not in decision.reasons


def test_limited_skill_overage_worsening_is_rejected() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "거북이는 오솔길을 따라 천천히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(
        skills=(SkillPolicy("HAS_TENSE_ONSET", "LIMITED", max_occurrences=1),)
    )

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"HAS_TENSE_ONSET": 1},
        proposal_occurrences={"HAS_TENSE_ONSET": 2},
        changed=(3,),
        context=_context(child_input=""),
        source_risk=2.0,
        proposal_risk=1.0,
    )

    assert decision.accepted is False
    assert "SKILL_WORSENED:HAS_TENSE_ONSET" in decision.reasons


def test_real_feature_improvement_with_preserved_meaning_is_accepted() -> None:
    source = _candidate()
    sentences = list(source.sentences)
    sentences[2] = "거북이는 돌길을 차분히 걸어가요."
    proposal = _candidate(
        candidate_id="proposal",
        sentences=tuple(sentences),  # type: ignore[arg-type]
    )
    profile = GenerationProfile(skills=(SkillPolicy("CODA_ㅆ", "EXCLUDED", max_occurrences=0),))

    decision = _decision(
        source_candidate=source,
        proposal_candidate=proposal,
        profile=profile,
        source_occurrences={"CODA_ㅆ": 2},
        proposal_occurrences={"CODA_ㅆ": 1},
        changed=(3,),
    )

    assert decision.accepted is True
    assert decision.reasons == ()
    assert "SKILL_OVERAGE_REDUCED:CODA_ㅆ" in decision.improvements
    assert "NORMALIZED_RISK_REDUCED" in decision.improvements
