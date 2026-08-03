from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from iread_ai.personalization.analyzer import (
    AnalysisStatus,
    CandidateAnalysis,
    KoreanReadingAnalyzer,
)
from iread_ai.personalization.generator import (
    PageCandidate,
    PageGenerationContext,
    RepairBatch,
)
from iread_ai.personalization.hangul import written_syllable_count
from iread_ai.personalization.selector import (
    CandidateEvaluation,
    GenerationProfile,
    SkillPolicy,
)

_CURLY_DIALOGUE_PATTERN = re.compile(r"“[^“”\r\n]+[.!?]”")
_ANY_DIALOGUE_PATTERN = re.compile(r'("[^"\r\n]+"|“[^”\r\n]+”|‘[^’\r\n]+’)')
_REPORTING_VERB_PATTERN = re.compile(
    r"(말(?:해요|했어요|하지요|했지요)|"
    r"외(?:쳐요|쳤어요)|"
    r"물(?:어요|었어요)|"
    r"대답(?:해요|했어요)|"
    r"속삭(?:여요|였어요)|"
    r"되뇌(?:어요|었어요)|"
    r"중얼(?:거려요|거렸어요)|"
    r"노래(?:해요|했어요)|"
    r"응원(?:해요|했어요)|"
    r"소리(?:쳐요|쳤어요))"
)
_HANGUL_WORD_PATTERN = re.compile(r"[가-힣]+")
_META_CHILD_REFERENCE_PATTERN = re.compile(
    r"(?<![가-힣])(?:"
    r"아이(?:가|는|의|를|에게|와|랑|도|들이|들은|들을)?|"
    r"독자(?:가|는|의|를|에게|와|도)?|"
    r"사용자(?:가|는|의|를|에게|와|도)?"
    r")(?![가-힣])"
)
_LIMITED_OVERAGE_REPAIR_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class RepairDecision:
    accepted: bool
    reasons: tuple[str, ...]
    improvements: tuple[str, ...]
    per_skill: tuple[dict[str, Any], ...]
    semantic_overlap: tuple[dict[str, Any], ...]
    source_speaker: str | None
    proposal_speaker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "improvements": list(self.improvements),
            "perSkill": [dict(row) for row in self.per_skill],
            "semanticOverlap": [dict(row) for row in self.semantic_overlap],
            "sourceSpeaker": self.source_speaker,
            "proposalSpeaker": self.proposal_speaker,
        }


def build_repair_plan(
    candidate: PageCandidate,
    evaluation: CandidateEvaluation,
    profile: GenerationProfile,
    context: PageGenerationContext,
    analyzer: KoreanReadingAnalyzer,
) -> dict[str, Any]:
    sentence_analyses = tuple(
        analyzer.analyze(
            (sentence,),
            protected_terms=profile.protected_terms,
        )
        for sentence in candidate.sentences
    )
    violations: list[dict[str, Any]] = []
    trigger_reasons: list[str] = []
    edit_scores: Counter[int] = Counter()
    first_sentence_edit_allowed = False
    limited_excess_total = 0
    child_input_missing = (
        requires_child_input_reflection(context)
        and not child_input_signal_preserved(
            candidate.sentences[0],
            context.child_input,
        )
    )

    if child_input_missing:
        trigger_reasons.append("STORY:CHILD_INPUT_NOT_REFLECTED")
        edit_scores[1] += 150
        edit_scores[2] += 140
        first_sentence_edit_allowed = True
        violations.append(
            {
                "kind": "STORY",
                "code": "CHILD_INPUT_NOT_REFLECTED",
                "affected_sentence_indexes": [1, 2],
                "repair_hint": (
                    "1문장에서 아이 답의 짧은 핵심 표현을 실제 사건으로 만들고, "
                    "2문장에서 기존 등장인물이 그 사건에 구체적으로 반응하게 하세요."
                ),
            }
        )

    for failure in evaluation.contract_failures:
        trigger_reasons.append(f"CONTRACT:{failure}")
        affected: list[int] = []
        if failure == "DIRECT_DIALOGUE_COUNT":
            affected = _dialogue_sentence_indexes(candidate.sentences)
            if not affected:
                affected = [2]
            for index in affected:
                edit_scores[index] += 100
        elif failure == "WRITTEN_SYLLABLE_RANGE":
            affected = _length_repair_indexes(candidate, evaluation, profile)
            for index in affected:
                edit_scores[index] += 60
        else:
            affected = [2]
            edit_scores[2] += 80
        if 1 in affected and child_input_signal_preserved(
            candidate.sentences[0],
            context.child_input,
        ):
            first_sentence_edit_allowed = True
        violations.append(
            {
                "kind": "CONTRACT",
                "code": failure,
                "affected_sentence_indexes": affected,
                "repair_hint": _contract_repair_hint(failure),
            }
        )

    exact_dialogue = has_exact_spoken_dialogue(
        candidate.sentences,
        context.characters,
    )
    if not exact_dialogue:
        trigger_reasons.append("CONTRACT:CURLY_DIALOGUE_FORMAT")
        affected = _dialogue_sentence_indexes(candidate.sentences)
        if not affected:
            affected = [2]
        if 1 in affected and child_input_signal_preserved(
            candidate.sentences[0],
            context.child_input,
        ):
            first_sentence_edit_allowed = True
        for index in affected:
            edit_scores[index] += 95
        violations.append(
            {
                "kind": "CONTRACT",
                "code": "CURLY_DIALOGUE_FORMAT",
                "affected_sentence_indexes": affected,
                "repair_hint": (
                    "직접 대사는 “와 ”를 정확히 한 쌍 사용하고 "
                    "같은 문장에 등장인물 화자와 발화 동사를 밝히세요."
                ),
            }
        )

    if has_meta_child_reference(candidate.sentences, context.characters):
        affected = [
            index
            for index, sentence in enumerate(candidate.sentences, start=1)
            if _META_CHILD_REFERENCE_PATTERN.search(sentence)
        ]
        trigger_reasons.append("STORY:META_CHILD_REFERENCE")
        for index in affected:
            edit_scores[index] += 120
        if 1 in affected:
            first_sentence_edit_allowed = True
        violations.append(
            {
                "kind": "STORY",
                "code": "META_CHILD_REFERENCE",
                "affected_sentence_indexes": affected,
                "repair_hint": (
                    "아이, 독자, 사용자를 이야기 속 인물이나 화자로 쓰지 말고 "
                    "같은 뜻을 기존 등장인물의 대사나 이야기 속 응원으로 바꾸세요."
                ),
            }
        )

    for skill in profile.skills:
        if skill.role not in {"EXCLUDED", "LIMITED"}:
            continue
        actual = _reliable_occurrences(evaluation, skill)
        if actual is None:
            continue
        maximum = skill.max_occurrences or 0
        excess = max(0, actual - maximum)
        if not excess:
            continue

        kind = f"{skill.role}_OVERAGE"
        if skill.role == "EXCLUDED":
            trigger_reasons.append(f"{kind}:{skill.code}")
        else:
            limited_excess_total += excess
        evidence: list[dict[str, Any]] = []
        for sentence_index, analysis in enumerate(sentence_analyses, start=1):
            count = _analysis_occurrences(analysis, skill.code)
            if count is None or not count:
                continue
            evidence.append(
                {
                    "sentence_index": sentence_index,
                    "count": count,
                    "analysis_status": analysis.status.value,
                    "pronunciation": (
                        analysis.pronunciations[0]
                        if analysis.pronunciations
                        else None
                    ),
                }
            )
            edit_scores[sentence_index] += count * (
                20 if skill.role == "EXCLUDED" else 5
            )
        violations.append(
            {
                "kind": kind,
                "code": skill.code,
                "actual": actual,
                "expected_max": maximum,
                "excess": excess,
                "affected_sentence_indexes": [
                    row["sentence_index"] for row in evidence
                ],
                "evidence": evidence,
                "repair_hint": (
                    "사건과 화자를 유지하고 같은 뜻의 쉬운 표현으로 "
                    "해당 특징의 초과만 줄이세요."
                ),
            }
        )

    if limited_excess_total >= _LIMITED_OVERAGE_REPAIR_THRESHOLD:
        trigger_reasons.append(
            f"LIMITED_OVERAGE_TOTAL:{limited_excess_total}"
        )

    safe_scores = {
        index: score
        for index, score in edit_scores.items()
        if index != 1 or first_sentence_edit_allowed
    }
    ranked_indexes = [
        index
        for index, _ in sorted(
            safe_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    editable_indexes = (
        [1, *(index for index in ranked_indexes if index != 1)][:2]
        if child_input_missing
        else ranked_indexes[:2]
    )
    return {
        "editable_sentence_indexes": editable_indexes,
        "max_changed_sentences": 2,
        "first_sentence_edit_allowed": first_sentence_edit_allowed,
        "trigger_reasons": list(dict.fromkeys(trigger_reasons)),
        "violations": violations,
        "limited_overage_total": limited_excess_total,
        "limited_overage_repair_threshold": (
            _LIMITED_OVERAGE_REPAIR_THRESHOLD
        ),
        "sentence_evidence": [
            {
                "sentence_index": index,
                "written_syllables": analysis.written_syllables,
                "analysis_status": analysis.status.value,
                "pronunciation": (
                    analysis.pronunciations[0]
                    if analysis.pronunciations
                    else None
                ),
            }
            for index, analysis in enumerate(sentence_analyses, start=1)
        ],
    }


def apply_repair_batch(
    source: PageCandidate,
    repair_batch: RepairBatch,
) -> tuple[PageCandidate, tuple[int, ...]]:
    sentences = list(source.sentences)
    changed: list[int] = []
    for replacement in repair_batch.replacements:
        index = replacement.sentence_index
        if sentences[index - 1] == replacement.sentence:
            continue
        sentences[index - 1] = replacement.sentence
        changed.append(index)
    proposal = PageCandidate(
        candidate_id=f"{source.candidate_id}-repair",
        sentences=tuple(sentences),  # type: ignore[arg-type]
        question=source.question,
        choices=source.choices,
    )
    return proposal, tuple(sorted(changed))


def evaluate_repair(
    *,
    source_candidate: PageCandidate,
    source: CandidateEvaluation,
    proposal_candidate: PageCandidate,
    proposal: CandidateEvaluation,
    context: PageGenerationContext,
    profile: GenerationProfile,
    editable_indexes: tuple[int, ...],
    changed_sentence_numbers: tuple[int, ...],
) -> RepairDecision:
    reasons: list[str] = []
    improvements: list[str] = []
    changed = set(changed_sentence_numbers)
    allowed = set(editable_indexes)

    if not 1 <= len(changed) <= 2:
        reasons.append("CHANGED_SENTENCE_COUNT")
    if not changed.issubset(allowed):
        reasons.append("LOCKED_SENTENCE_CHANGED")
    for index, (before, after) in enumerate(
        zip(
            source_candidate.sentences,
            proposal_candidate.sentences,
            strict=True,
        ),
        start=1,
    ):
        if index not in changed and before != after:
            reasons.append(f"UNDECLARED_SENTENCE_CHANGE:{index}")

    missing_terms = _missing_protected_terms(
        source_candidate,
        proposal_candidate,
        context,
        profile,
    )
    if missing_terms:
        reasons.append("PROTECTED_TERM_LOST:" + ",".join(missing_terms))

    source_child_signal = child_input_signal_preserved(
        source_candidate.sentences[0],
        context.child_input,
    )
    proposal_child_signal = child_input_signal_preserved(
        proposal_candidate.sentences[0],
        context.child_input,
    )
    if requires_child_input_reflection(context):
        if not proposal_child_signal:
            reasons.append("CHILD_INPUT_SIGNAL_MISSING")
        elif not source_child_signal:
            improvements.append("CHILD_INPUT_SIGNAL_ADDED")

    if not _page_child_signal_preserved(
        source_candidate.sentences,
        proposal_candidate.sentences,
        context.child_input,
    ):
        reasons.append("CHILD_INPUT_SIGNAL_LOST")

    source_speaker = _dialogue_speaker(
        source_candidate.sentences,
        context.characters,
    )
    proposal_speaker = _dialogue_speaker(
        proposal_candidate.sentences,
        context.characters,
    )
    if not has_exact_spoken_dialogue(
        proposal_candidate.sentences,
        context.characters,
    ):
        reasons.append("DIALOGUE_FORMAT_OR_SPEAKER")
    if source_speaker is not None and proposal_speaker != source_speaker:
        reasons.append("DIALOGUE_SPEAKER_CHANGED")
    if _dialogue_is_duplicated(proposal_candidate.sentences):
        reasons.append("DIALOGUE_DUPLICATED")
    source_has_meta_reference = has_meta_child_reference(
        source_candidate.sentences,
        context.characters,
    )
    proposal_has_meta_reference = has_meta_child_reference(
        proposal_candidate.sentences,
        context.characters,
    )
    if proposal_has_meta_reference:
        reasons.append("META_CHILD_REFERENCE_REMAINS")
    elif source_has_meta_reference:
        improvements.append("META_CHILD_REFERENCE_REMOVED")

    semantic_rows: list[dict[str, Any]] = []
    for sentence_number in sorted(changed):
        before = source_candidate.sentences[sentence_number - 1]
        after = proposal_candidate.sentences[sentence_number - 1]
        overlap = _semantic_overlap(before, after)
        polarity_preserved = _negation_polarity(before) == _negation_polarity(after)
        source_characters = _character_set(
            before,
            context.characters,
        )
        proposal_characters = _character_set(after, context.characters)
        character_set_preserved = (
            source_characters == proposal_characters
            or _safe_missing_speaker_annotation(
                before=before,
                after=after,
                source_characters=source_characters,
                proposal_characters=proposal_characters,
                source_speaker=source_speaker,
                proposal_speaker=proposal_speaker,
            )
        )
        row = {
            "sentenceNumber": sentence_number,
            "overlap": round(overlap, 6),
            "polarityPreserved": polarity_preserved,
            "characterSetPreserved": character_set_preserved,
        }
        semantic_rows.append(row)
        if overlap < 0.28:
            reasons.append(f"SEMANTIC_OVERLAP_LOW:{sentence_number}")
        if not polarity_preserved:
            reasons.append(f"NEGATION_CHANGED:{sentence_number}")
        if not character_set_preserved:
            reasons.append(f"CHARACTER_ROLE_CHANGED:{sentence_number}")

    source_failures = set(source.contract_failures)
    proposal_failures = set(proposal.contract_failures)
    if not proposal_failures.issubset(source_failures):
        reasons.append("NEW_CONTRACT_FAILURE")
    if proposal.contract_penalty > source.contract_penalty:
        reasons.append("CONTRACT_PENALTY_WORSENED")
    if len(proposal_failures) < len(source_failures):
        improvements.append("CONTRACT_FAILURE_REDUCED")
    elif proposal.contract_penalty < source.contract_penalty:
        improvements.append("CONTRACT_PENALTY_REDUCED")

    if _analysis_rank(proposal.analysis.status) > _analysis_rank(
        source.analysis.status
    ):
        reasons.append("G2P_STATUS_WORSENED")
    elif _analysis_rank(proposal.analysis.status) < _analysis_rank(
        source.analysis.status
    ):
        improvements.append("G2P_STATUS_IMPROVED")

    per_skill: list[dict[str, Any]] = []
    for skill in profile.skills:
        source_count = _reliable_occurrences(source, skill)
        proposal_count = _reliable_occurrences(proposal, skill)
        row: dict[str, Any] = {
            "code": skill.code,
            "role": skill.role,
            "source": source_count,
            "proposal": proposal_count,
        }
        if source_count is None or proposal_count is None:
            row["comparable"] = False
            if skill.code.startswith("PHONO_") and (
                source_count != proposal_count
                or changed
            ):
                row["reason"] = "G2P_NOT_FULL"
            per_skill.append(row)
            continue

        row["comparable"] = True
        if skill.role in {"EXCLUDED", "LIMITED"}:
            maximum = skill.max_occurrences or 0
            source_overage = max(0, source_count - maximum)
            proposal_overage = max(0, proposal_count - maximum)
            row["sourceOverage"] = source_overage
            row["proposalOverage"] = proposal_overage
            worsened = (
                proposal_overage > source_overage
                if skill.role == "LIMITED"
                else (
                    proposal_count > source_count
                    or proposal_overage > source_overage
                )
            )
            if worsened:
                reasons.append(f"SKILL_WORSENED:{skill.code}")
            if proposal_overage < source_overage:
                improvements.append(f"SKILL_OVERAGE_REDUCED:{skill.code}")
        elif skill.role == "TARGET":
            source_distance = _target_distance(source_count, skill)
            proposal_distance = _target_distance(proposal_count, skill)
            row["sourceDistance"] = source_distance
            row["proposalDistance"] = proposal_distance
            if proposal_distance > source_distance:
                reasons.append(f"TARGET_WORSENED:{skill.code}")
            if proposal_distance < source_distance:
                improvements.append(f"TARGET_IMPROVED:{skill.code}")
        per_skill.append(row)

    if proposal.risk_per_10 > source.risk_per_10 + 1e-9:
        reasons.append("NORMALIZED_RISK_WORSENED")
    elif proposal.risk_per_10 < source.risk_per_10 - 1e-9:
        improvements.append("NORMALIZED_RISK_REDUCED")

    if (
        not has_exact_spoken_dialogue(
            source_candidate.sentences,
            context.characters,
        )
        and has_exact_spoken_dialogue(
            proposal_candidate.sentences,
            context.characters,
        )
    ):
        improvements.append("DIALOGUE_FIXED")
    if not improvements:
        reasons.append("NO_MEASURABLE_IMPROVEMENT")

    return RepairDecision(
        accepted=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        improvements=tuple(dict.fromkeys(improvements)),
        per_skill=tuple(per_skill),
        semantic_overlap=tuple(semantic_rows),
        source_speaker=source_speaker,
        proposal_speaker=proposal_speaker,
    )


def has_exact_spoken_dialogue(
    sentences: tuple[str, ...],
    characters: tuple[str, ...] = (),
) -> bool:
    text = "\n".join(sentences)
    matches: list[tuple[str, re.Match[str]]] = []
    for sentence in sentences:
        match = _CURLY_DIALOGUE_PATTERN.search(sentence)
        if match is not None:
            matches.append((sentence, match))
    if not (
        text.count("“") == 1
        and text.count("”") == 1
        and '"' not in text
        and "‘" not in text
        and "’" not in text
        and len(matches) == 1
    ):
        return False
    sentence, match = matches[0]
    outside = sentence[: match.start()] + sentence[match.end() :]
    if _REPORTING_VERB_PATTERN.search(outside) is None:
        return False
    if characters:
        return any(character in outside for character in characters)
    return bool(re.search(r"[가-힣]{1,12}(?:은|는|이|가)", outside))


def has_hard_repair_trigger(repair_plan: dict[str, Any]) -> bool:
    return bool(repair_plan.get("trigger_reasons"))


def has_meta_child_reference(
    sentences: tuple[str, ...],
    characters: tuple[str, ...] = (),
) -> bool:
    if any(character.strip() == "아이" for character in characters):
        return False
    return any(
        _META_CHILD_REFERENCE_PATTERN.search(sentence) is not None
        for sentence in sentences
    )


def _dialogue_sentence_indexes(sentences: tuple[str, ...]) -> list[int]:
    return [
        index
        for index, sentence in enumerate(sentences, start=1)
        if _ANY_DIALOGUE_PATTERN.search(sentence)
    ]


def _length_repair_indexes(
    candidate: PageCandidate,
    evaluation: CandidateEvaluation,
    profile: GenerationProfile,
) -> list[int]:
    rows = [
        (index, written_syllable_count(sentence))
        for index, sentence in enumerate(candidate.sentences, start=1)
        if index != 1
    ]
    if evaluation.analysis.written_syllables > (
        profile.content_contract.accepted_max_syllables
    ):
        rows.sort(key=lambda row: (-row[1], row[0]))
    else:
        rows.sort(key=lambda row: (row[1], row[0]))
    return [index for index, _ in rows[:2]]


def _contract_repair_hint(code: str) -> str:
    return {
        "SENTENCE_COUNT": "원문의 문장 수를 바꾸지 말고 그대로 유지하세요.",
        "BLANK_SENTENCE": "빈 문장을 현재 사건을 유지하는 자연스러운 문장으로 채우세요.",
        "WRITTEN_SYLLABLE_RANGE": (
            "행동과 인과를 바꾸지 않고 전체 한글 음절을 허용 범위에 맞추세요."
        ),
        "DIRECT_DIALOGUE_COUNT": (
            "기존 화자와 의미를 유지해 직접 대사 문장을 정확히 하나로 만드세요."
        ),
    }.get(code, "현재 사건의 의미를 유지하면서 해당 계약만 고치세요.")


def _analysis_occurrences(
    analysis: CandidateAnalysis,
    code: str,
) -> int | None:
    if code.startswith("PHONO_"):
        if analysis.status is not AnalysisStatus.FULL:
            return None
        return int(analysis.phonological_rule_counts.get(code, 0))
    return int(analysis.controllable_surface_feature_counts.get(code, 0))


def _reliable_occurrences(
    evaluation: CandidateEvaluation,
    skill: SkillPolicy,
) -> int | None:
    if (
        skill.code.startswith("PHONO_")
        and evaluation.analysis.status is not AnalysisStatus.FULL
    ):
        return None
    return int(evaluation.feature_occurrences.get(skill.code, 0))


def _target_distance(count: int, skill: SkillPolicy) -> int:
    minimum = skill.target_min if skill.target_min is not None else count
    maximum = skill.target_max if skill.target_max is not None else count
    if count < minimum:
        return minimum - count
    if count > maximum:
        return count - maximum
    return 0


def _missing_protected_terms(
    source: PageCandidate,
    proposal: PageCandidate,
    context: PageGenerationContext,
    profile: GenerationProfile,
) -> list[str]:
    source_text = " ".join(source.sentences)
    proposal_text = " ".join(proposal.sentences)
    terms = tuple(dict.fromkeys((*context.characters, *profile.protected_terms)))
    return [
        term
        for term in terms
        if source_text.count(term) > proposal_text.count(term)
    ]


def _dialogue_speaker(
    sentences: tuple[str, ...],
    characters: tuple[str, ...],
) -> str | None:
    for sentence in sentences:
        match = _CURLY_DIALOGUE_PATTERN.search(sentence)
        if match is None:
            continue
        outside = sentence[: match.start()] + sentence[match.end() :]
        if _REPORTING_VERB_PATTERN.search(outside) is None:
            return None
        for character in characters:
            if re.search(
                rf"{re.escape(character)}(?:은|는|이|가)",
                outside,
            ):
                return character
        speaker_match = re.search(r"([가-힣]{1,12})(?:은|는|이|가)", outside)
        if speaker_match is not None:
            return speaker_match.group(1)
    return None


def _character_set(text: str, characters: tuple[str, ...]) -> frozenset[str]:
    return frozenset(character for character in characters if character in text)


def _safe_missing_speaker_annotation(
    *,
    before: str,
    after: str,
    source_characters: frozenset[str],
    proposal_characters: frozenset[str],
    source_speaker: str | None,
    proposal_speaker: str | None,
) -> bool:
    if source_speaker is not None or proposal_speaker is None:
        return False
    if _ANY_DIALOGUE_PATTERN.search(before) is None:
        return False
    if _ANY_DIALOGUE_PATTERN.search(after) is None:
        return False
    return proposal_characters == source_characters | {proposal_speaker}


def _negation_polarity(text: str) -> bool:
    compact = " ".join(_HANGUL_WORD_PATTERN.findall(text))
    return bool(re.search(r"(?:^|\s)(?:안|못)(?:\s|$)|않|없", compact))


def _semantic_overlap(source: str, proposal: str) -> float:
    source_bigrams = _hangul_bigrams(source)
    proposal_bigrams = _hangul_bigrams(proposal)
    if not source_bigrams or not proposal_bigrams:
        return 0.0
    return 2.0 * len(source_bigrams & proposal_bigrams) / (
        len(source_bigrams) + len(proposal_bigrams)
    )


def requires_child_input_reflection(context: PageGenerationContext) -> bool:
    return context.page_number == 1 and bool(context.child_input.strip())


def child_input_signal_preserved(sentence: str, child_input: str) -> bool:
    sentence_hangul = "".join(_HANGUL_WORD_PATTERN.findall(sentence))
    child_hangul = "".join(_HANGUL_WORD_PATTERN.findall(child_input))
    if len(child_hangul) == 1:
        return child_hangul in sentence_hangul
    child_phrases = {
        phrase
        for phrase in _HANGUL_WORD_PATTERN.findall(child_input)
        if len(phrase) >= 3
    }
    if any(phrase in sentence_hangul for phrase in child_phrases):
        return True
    child_bigrams = _hangul_bigrams(child_input)
    if not child_bigrams:
        return False
    sentence_bigrams = _hangul_bigrams(sentence)
    return len(child_bigrams & sentence_bigrams) / len(child_bigrams) >= 0.4


def _page_child_signal_preserved(
    source: tuple[str, ...],
    proposal: tuple[str, ...],
    child_input: str,
) -> bool:
    child_bigrams = _hangul_bigrams(child_input)
    if not child_bigrams:
        return True
    source_signal = child_bigrams & _hangul_bigrams(" ".join(source))
    if not source_signal:
        return True
    proposal_signal = child_bigrams & _hangul_bigrams(" ".join(proposal))
    return len(source_signal & proposal_signal) / len(source_signal) >= 0.8


def _dialogue_is_duplicated(sentences: tuple[str, ...]) -> bool:
    dialogue_index: int | None = None
    dialogue_text = ""
    for index, sentence in enumerate(sentences):
        match = _CURLY_DIALOGUE_PATTERN.search(sentence)
        if match is not None:
            dialogue_index = index
            dialogue_text = match.group(0)[1:-1]
            break
    if dialogue_index is None:
        return False
    dialogue_bigrams = _hangul_bigrams(dialogue_text)
    if not dialogue_bigrams:
        return False
    for index, sentence in enumerate(sentences):
        if index == dialogue_index:
            continue
        other = _hangul_bigrams(sentence)
        if other and len(dialogue_bigrams & other) / len(dialogue_bigrams) >= 0.7:
            return True
    return False


def _hangul_bigrams(text: str) -> set[str]:
    hangul = "".join(_HANGUL_WORD_PATTERN.findall(text))
    if len(hangul) < 2:
        return {hangul} if hangul else set()
    return {hangul[index : index + 2] for index in range(len(hangul) - 1)}


def _analysis_rank(status: AnalysisStatus) -> int:
    return {
        AnalysisStatus.FULL: 0,
        AnalysisStatus.UNRELIABLE: 1,
        AnalysisStatus.SURFACE_ONLY: 2,
    }[status]


__all__ = [
    "RepairDecision",
    "apply_repair_batch",
    "build_repair_plan",
    "child_input_signal_preserved",
    "evaluate_repair",
    "has_exact_spoken_dialogue",
    "has_hard_repair_trigger",
    "has_meta_child_reference",
    "requires_child_input_reflection",
]
