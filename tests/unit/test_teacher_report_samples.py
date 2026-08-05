from __future__ import annotations

from iread_ai.application.teacher_report_analyzer import TeacherReportAnalyzer
from iread_ai.contracts.teacher_report import TeacherReportAnalyzeRequest
from iread_ai.devtools.teacher_report_samples import get_teacher_report_samples


def _analyze(sample_id: str):
    sample = next(
        sample for sample in get_teacher_report_samples() if sample.sample_id == sample_id
    )
    request = TeacherReportAnalyzeRequest.model_validate(sample.payload)
    return TeacherReportAnalyzer().analyze(request)


def test_every_teacher_report_sample_matches_the_request_contract() -> None:
    samples = get_teacher_report_samples()

    requests = [
        TeacherReportAnalyzeRequest.model_validate(sample.payload) for sample in samples
    ]

    assert len(samples) == 4
    assert len({sample.sample_id for sample in samples}) == len(samples)
    assert len({request.request_id for request in requests}) == len(requests)


def test_balanced_progress_exercises_all_main_result_sections() -> None:
    facts = _analyze("balanced-progress")

    assert facts.data_sufficiency == "SUFFICIENT"
    assert facts.improved
    assert facts.persistent
    assert any(fact.direction == "decrease" for fact in facts.test_gaze)


def test_effortful_success_detects_hidden_reading_burden() -> None:
    facts = _analyze("effortful-success")

    assert facts.data_sufficiency == "PARTIAL"
    assert any(fact.direction == "effort" for fact in facts.persistent)


def test_insufficient_sample_withholds_feature_judgment() -> None:
    facts = _analyze("insufficient-evidence")

    assert facts.data_sufficiency == "INSUFFICIENT"
    assert facts.improved == ()
    assert facts.persistent == ()
    assert facts.training_gaze[0].direction == "unavailable"


def test_failed_gaze_sample_withholds_gaze_interpretation() -> None:
    facts = _analyze("failed-gaze")

    assert facts.training_gaze[0].direction == "unavailable"
    assert "실패한 세션 2건" in facts.training_gaze[0].text


def test_sample_factory_returns_independent_payloads() -> None:
    first = get_teacher_report_samples()
    first[0].payload["requestId"] = "changed"

    second = get_teacher_report_samples()

    assert second[0].payload["requestId"] == "demo-balanced-progress"
