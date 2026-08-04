from __future__ import annotations

from dataclasses import dataclass, field

from iread_ai.application.teacher_report_summary_service import (
    TeacherReportSummaryService,
)
from iread_ai.contracts.teacher_report import (
    TeacherReportAnalyzeRequest,
    TeacherReportNarrativeDraft,
)
from iread_ai.ports.teacher_report_narrator import EvidenceStatement
from tests.unit.test_teacher_report_contracts import teacher_report_request_payload


@dataclass
class EvidenceEchoNarrator:
    calls: list[tuple[EvidenceStatement, ...]] = field(default_factory=list)
    replacement_text: str | None = None

    @property
    def provider_name(self) -> str:
        return "gms"

    def narrate(
        self,
        evidence: tuple[EvidenceStatement, ...],
    ) -> TeacherReportNarrativeDraft:
        self.calls.append(evidence)
        by_category = {
            category: [item for item in evidence if item.category == category]
            for category in (
                "improved",
                "persistent",
                "training_gaze",
                "test_gaze",
            )
        }

        def statement(category: str) -> list[dict[str, object]]:
            item = by_category[category][0]
            return [
                {
                    "text": self.replacement_text or item.canonical_text,
                    "evidenceIds": [item.evidence_id],
                }
            ]

        return TeacherReportNarrativeDraft.model_validate(
            {
                "improvedPatterns": statement("improved"),
                "persistentDifficultyPatterns": statement("persistent"),
                "trainingGazeDescriptions": statement("training_gaze"),
                "testGazeDescriptions": statement("test_gaze"),
            }
        )


class FailingNarrator:
    @property
    def provider_name(self) -> str:
        return "gms"

    def narrate(
        self,
        evidence: tuple[EvidenceStatement, ...],
    ) -> TeacherReportNarrativeDraft:
        del evidence
        raise RuntimeError("provider failed")


class GuidanceOmittingNarrator(EvidenceEchoNarrator):
    def narrate(
        self,
        evidence: tuple[EvidenceStatement, ...],
    ) -> TeacherReportNarrativeDraft:
        self.calls.append(evidence)
        items: dict[str, list[dict[str, object]]] = {
            "improvedPatterns": [],
            "persistentDifficultyPatterns": [],
            "trainingGazeDescriptions": [],
            "testGazeDescriptions": [],
        }
        key_by_category = {
            "improved": "improvedPatterns",
            "persistent": "persistentDifficultyPatterns",
            "training_gaze": "trainingGazeDescriptions",
            "test_gaze": "testGazeDescriptions",
        }
        for item in evidence:
            text = item.canonical_text
            if item.category == "persistent":
                text = f"{item.subject}에서 반복적인 어려움이 관찰됩니다."
            elif item.category in {"training_gaze", "test_gaze"} and "감소" in text:
                text = f"{item.subject}의 역행 읽기 횟수가 4회에서 2회로 감소했습니다."
            items[key_by_category[item.category]].append(
                {"text": text, "evidenceIds": [item.evidence_id]}
            )
        return TeacherReportNarrativeDraft.model_validate(items)


def _request() -> TeacherReportAnalyzeRequest:
    return TeacherReportAnalyzeRequest.model_validate(teacher_report_request_payload())


async def test_service_uses_evidence_backed_gms_narrative() -> None:
    narrator = EvidenceEchoNarrator()
    service = TeacherReportSummaryService(narrator=narrator)

    response = await service.analyze(_request())

    assert response.summary_provider == "gms"
    assert response.analysis_version == "TEACHER_REPORT_ANALYSIS_V1"
    assert response.improved_patterns[0].startswith("초성 ㄱ 읽기")
    assert response.gaze_descriptions.test[0].startswith("검사 시선")
    assert len(narrator.calls) == 1


async def test_service_falls_back_when_narrator_introduces_a_diagnosis() -> None:
    narrator = EvidenceEchoNarrator(replacement_text="초성 ㄱ 읽기에서 난독증 진단이 확실합니다.")
    service = TeacherReportSummaryService(narrator=narrator)

    response = await service.analyze(_request())

    assert response.summary_provider == "deterministic-fallback"
    rendered = " ".join(
        response.improved_patterns
        + response.persistent_difficulty_patterns
        + response.gaze_descriptions.training
        + response.gaze_descriptions.test
    )
    assert "난독증" not in rendered
    assert "진단" not in rendered


async def test_service_falls_back_when_narrator_invents_a_number() -> None:
    narrator = EvidenceEchoNarrator(replacement_text="초성 ㄱ 읽기 정확도가 99%로 관찰됩니다.")
    service = TeacherReportSummaryService(narrator=narrator)

    response = await service.analyze(_request())

    assert response.summary_provider == "deterministic-fallback"
    assert "99%" not in " ".join(response.improved_patterns)


async def test_service_falls_back_when_narrator_exposes_evidence_id() -> None:
    narrator = EvidenceEchoNarrator(
        replacement_text=(
            "초성 ㄱ 읽기 정확도가 관찰됩니다. "
            "[evidenceId: improved-c5f39085b390b1a5]"
        )
    )
    service = TeacherReportSummaryService(narrator=narrator)

    response = await service.analyze(_request())

    assert response.summary_provider == "deterministic-fallback"
    rendered = " ".join(
        response.improved_patterns
        + response.persistent_difficulty_patterns
        + response.gaze_descriptions.training
        + response.gaze_descriptions.test
    )
    assert "evidenceId" not in rendered


async def test_service_falls_back_on_provider_failure() -> None:
    response = await TeacherReportSummaryService(narrator=FailingNarrator()).analyze(_request())

    assert response.summary_provider == "deterministic-fallback"
    assert response.improved_patterns
    assert response.persistent_difficulty_patterns


async def test_service_falls_back_when_narrator_omits_follow_up_guidance() -> None:
    response = await TeacherReportSummaryService(
        narrator=GuidanceOmittingNarrator()
    ).analyze(_request())

    assert response.summary_provider == "deterministic-fallback"
    assert "다음 회기에서도 지속 관찰" in response.persistent_difficulty_patterns[0]
    assert "추가 확인이 필요" in response.gaze_descriptions.training[0]


async def test_service_is_deterministic_without_configured_gms() -> None:
    response = await TeacherReportSummaryService().analyze(_request())

    assert response.summary_provider == "deterministic"
    assert response.data_sufficiency == "SUFFICIENT"
