from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from iread_ai.contracts.teacher_report import TeacherReportNarrativeDraft

EvidenceCategory = Literal["improved", "persistent", "training_gaze", "test_gaze"]


@dataclass(frozen=True, slots=True)
class EvidenceStatement:
    evidence_id: str
    category: EvidenceCategory
    subject: str
    canonical_text: str


class TeacherReportNarratorError(RuntimeError):
    """Safe narrator failure without prompts, child data, or credentials."""


class TeacherReportNarrator(Protocol):
    @property
    def provider_name(self) -> str: ...

    def narrate(
        self,
        evidence: tuple[EvidenceStatement, ...],
    ) -> TeacherReportNarrativeDraft: ...


__all__ = [
    "EvidenceStatement",
    "TeacherReportNarrator",
    "TeacherReportNarratorError",
]
