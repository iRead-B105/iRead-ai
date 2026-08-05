from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

from iread_ai.application.teacher_report_analyzer import (
    TeacherReportAnalyzer,
    TeacherReportFact,
    TeacherReportFacts,
)
from iread_ai.contracts.teacher_report import (
    SummaryProvider,
    TeacherReportAnalyzeRequest,
    TeacherReportAnalyzeResponse,
    TeacherReportGazeDescriptions,
    TeacherReportNarrativeDraft,
    TeacherReportNarrativeItem,
)
from iread_ai.ports.teacher_report_narrator import (
    EvidenceStatement,
    TeacherReportNarrator,
)

logger = logging.getLogger(__name__)

PROHIBITED_EXPRESSIONS = (
    "난독증",
    "진단",
    "중증",
    "장애",
    "원인",
    "치료",
    "완치",
    "정상",
    "비정상",
    "확실",
    "반드시",
    "dyslexia",
    "diagnosis",
    "severity",
    "disorder",
    "treatment",
    "cure",
)
INCREASE_WORDS = ("증가", "늘었", "높아졌", "상승")
DECREASE_WORDS = ("감소", "줄었", "낮아졌", "하락")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")


class TeacherReportSummaryService:
    def __init__(
        self,
        *,
        analyzer: TeacherReportAnalyzer | None = None,
        narrator: TeacherReportNarrator | None = None,
    ) -> None:
        self._analyzer = analyzer or TeacherReportAnalyzer()
        self._narrator = narrator

    async def analyze(
        self,
        request: TeacherReportAnalyzeRequest,
    ) -> TeacherReportAnalyzeResponse:
        facts = self._analyzer.analyze(request)
        provider: SummaryProvider = "deterministic"
        narrative = self._deterministic_narrative(facts)

        if self._narrator is not None:
            try:
                draft = await asyncio.to_thread(
                    self._narrator.narrate,
                    _as_evidence(facts),
                )
                narrative = self._validate_narrative(draft, facts)
                provider = "gms"
            except Exception as exception:
                provider = "deterministic-fallback"
                logger.warning(
                    "Teacher report narrator fallback request_id=%s error=%s",
                    request.request_id,
                    type(exception).__name__,
                )

        response = TeacherReportAnalyzeResponse(
            requestId=request.request_id,
            schemaVersion=1,
            analysisVersion=self._analyzer.analysis_version,
            summaryProvider=provider,
            dataSufficiency=facts.data_sufficiency,
            improvedPatterns=narrative.improved_patterns,
            persistentDifficultyPatterns=narrative.persistent_difficulty_patterns,
            gazeDescriptions=TeacherReportGazeDescriptions(
                training=narrative.training_gaze_descriptions,
                test=narrative.test_gaze_descriptions,
            ),
        )
        return response.validate_against_request(request)

    def _deterministic_narrative(
        self,
        facts: TeacherReportFacts,
    ) -> _RenderedNarrative:
        return _RenderedNarrative(
            improved_patterns=tuple(fact.text for fact in facts.improved),
            persistent_difficulty_patterns=tuple(fact.text for fact in facts.persistent),
            training_gaze_descriptions=tuple(fact.text for fact in facts.training_gaze),
            test_gaze_descriptions=tuple(fact.text for fact in facts.test_gaze),
        )

    def _validate_narrative(
        self,
        draft: TeacherReportNarrativeDraft,
        facts: TeacherReportFacts,
    ) -> _RenderedNarrative:
        evidence = {fact.evidence_id: fact for fact in facts.all}
        used_ids: set[str] = set()
        improved = _validate_items(
            draft.improved_patterns,
            expected_category="improved",
            available_facts=facts.improved,
            evidence=evidence,
            used_ids=used_ids,
        )
        persistent = _validate_items(
            draft.persistent_difficulty_patterns,
            expected_category="persistent",
            available_facts=facts.persistent,
            evidence=evidence,
            used_ids=used_ids,
        )
        training_gaze = _validate_items(
            draft.training_gaze_descriptions,
            expected_category="training_gaze",
            available_facts=facts.training_gaze,
            evidence=evidence,
            used_ids=used_ids,
        )
        test_gaze = _validate_items(
            draft.test_gaze_descriptions,
            expected_category="test_gaze",
            available_facts=facts.test_gaze,
            evidence=evidence,
            used_ids=used_ids,
        )
        return _RenderedNarrative(
            improved_patterns=improved,
            persistent_difficulty_patterns=persistent,
            training_gaze_descriptions=training_gaze,
            test_gaze_descriptions=test_gaze,
        )


class _RenderedNarrative:
    def __init__(
        self,
        *,
        improved_patterns: Sequence[str],
        persistent_difficulty_patterns: Sequence[str],
        training_gaze_descriptions: Sequence[str],
        test_gaze_descriptions: Sequence[str],
    ) -> None:
        self.improved_patterns = list(improved_patterns)
        self.persistent_difficulty_patterns = list(persistent_difficulty_patterns)
        self.training_gaze_descriptions = list(training_gaze_descriptions)
        self.test_gaze_descriptions = list(test_gaze_descriptions)


def _as_evidence(facts: TeacherReportFacts) -> tuple[EvidenceStatement, ...]:
    return tuple(
        EvidenceStatement(
            evidence_id=fact.evidence_id,
            category=fact.category,
            subject=fact.subject,
            canonical_text=fact.text,
        )
        for fact in facts.all
    )


def _validate_items(
    items: Sequence[TeacherReportNarrativeItem],
    *,
    expected_category: str,
    available_facts: Sequence[TeacherReportFact],
    evidence: dict[str, TeacherReportFact],
    used_ids: set[str],
) -> tuple[str, ...]:
    if available_facts and not items:
        raise ValueError(f"narrative omitted {expected_category} evidence")
    rendered: list[str] = []
    for item in items:
        cited: list[TeacherReportFact] = []
        for evidence_id in item.evidence_ids:
            fact = evidence.get(evidence_id)
            if fact is None or fact.category != expected_category:
                raise ValueError("narrative cited missing or cross-category evidence")
            if evidence_id in used_ids:
                raise ValueError("narrative reused evidence across statements")
            cited.append(fact)
        _validate_text(item.text, cited)
        used_ids.update(item.evidence_ids)
        rendered.append(item.text)
    return tuple(rendered)


def _validate_text(text: str, cited: Sequence[TeacherReportFact]) -> None:
    lowered = text.casefold()
    if any(expression in lowered for expression in PROHIBITED_EXPRESSIONS):
        raise ValueError("narrative contains a prohibited diagnostic expression")
    if "evidenceid" in lowered or any(fact.evidence_id in text for fact in cited):
        raise ValueError("narrative exposed an internal evidence identifier")
    if not any(fact.subject in text for fact in cited):
        raise ValueError("narrative does not identify its evidence subject")

    source_numbers = {number for fact in cited for number in NUMBER_PATTERN.findall(fact.text)}
    rendered_numbers = set(NUMBER_PATTERN.findall(text))
    if not rendered_numbers.issubset(source_numbers):
        raise ValueError("narrative introduced an unsupported number")

    directions = {fact.direction for fact in cited}
    if directions == {"increase"} and any(word in text for word in DECREASE_WORDS):
        raise ValueError("narrative reversed an increasing gaze trend")
    if directions == {"decrease"} and any(word in text for word in INCREASE_WORDS):
        raise ValueError("narrative reversed a decreasing gaze trend")
    if directions == {"improved"} and any(word in text for word in ("악화", "하락")):
        raise ValueError("narrative reversed an improvement fact")
    if directions == {"improved"} and not source_numbers.issubset(rendered_numbers):
        raise ValueError("narrative omitted an improvement comparison value")
    if directions & {"persistent", "effort"} and not source_numbers.issubset(
        rendered_numbers
    ):
        raise ValueError("narrative omitted a persistent observation value")
    if directions & {"persistent", "effort"} and not (
        "지속 관찰" in text or "지속해서 살펴" in text
    ):
        raise ValueError("narrative omitted continued observation guidance")
    if directions & {"increase", "decrease", "stable"} and not (
        "추가 확인" in text or "추가 관찰" in text
    ):
        raise ValueError("narrative omitted gaze follow-up guidance")


__all__ = ["TeacherReportSummaryService"]
