from __future__ import annotations

import json

import httpx
import pytest

from iread_ai.adapters.generation.gms_teacher_report import (
    GMSTeacherReportNarrator,
)
from iread_ai.ports.teacher_report_narrator import (
    EvidenceStatement,
    TeacherReportNarratorError,
)
from iread_ai.providers import GMSTextProvider


def _evidence() -> tuple[EvidenceStatement, ...]:
    return (
        EvidenceStatement(
            evidence_id="improved-a1",
            category="improved",
            subject="초성 읽기",
            canonical_text="초성 읽기 정확도가 60%에서 80%로 상승했습니다.",
        ),
        EvidenceStatement(
            evidence_id="persistent-b1",
            category="persistent",
            subject="받침 읽기",
            canonical_text="받침 읽기 정확도 45%로 추가 확인이 필요합니다.",
        ),
        EvidenceStatement(
            evidence_id="training-gaze-c1",
            category="training_gaze",
            subject="훈련 시선",
            canonical_text="훈련 시선의 역행 읽기가 4회에서 2회로 감소했습니다.",
        ),
        EvidenceStatement(
            evidence_id="test-gaze-d1",
            category="test_gaze",
            subject="검사 시선",
            canonical_text="검사 시선 데이터가 없어 변화 해석을 보류합니다.",
        ),
    )


def _output() -> dict[str, object]:
    return {
        "improvedPatterns": [
            {
                "text": "초성 읽기 정확도가 60%에서 80%로 상승했습니다.",
                "evidenceIds": ["improved-a1"],
            }
        ],
        "persistentDifficultyPatterns": [
            {
                "text": "받침 읽기 정확도 45%로 추가 확인이 필요합니다.",
                "evidenceIds": ["persistent-b1"],
            }
        ],
        "trainingGazeDescriptions": [
            {
                "text": "훈련 시선의 역행 읽기가 4회에서 2회로 감소했습니다.",
                "evidenceIds": ["training-gaze-c1"],
            }
        ],
        "testGazeDescriptions": [
            {
                "text": "검사 시선 데이터가 없어 변화 해석을 보류합니다.",
                "evidenceIds": ["test-gaze-d1"],
            }
        ],
    }


def _narrator(handler: httpx.MockTransport) -> GMSTeacherReportNarrator:
    client = httpx.Client(transport=handler)
    provider = GMSTextProvider(
        api_key="secret-gms-key",
        model="gpt-5.4-mini",
        base_url="https://gms.example/v1",
        timeout_seconds=1,
        max_output_tokens=512,
        client=client,
    )
    return GMSTeacherReportNarrator(provider)


def test_gms_narrator_uses_strict_schema_and_only_evidence_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output_text": json.dumps(_output(), ensure_ascii=False),
            },
        )

    draft = _narrator(httpx.MockTransport(handler)).narrate(_evidence())

    assert draft.improved_patterns[0].evidence_ids == ["improved-a1"]
    assert captured["store"] is False
    assert captured["model"] == "gpt-5.4-mini"
    text_format = captured["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    request_text = captured["input"][1]["content"][0]["text"]
    assert "canonicalText" in request_text
    assert "teacher-report-narrative-v3" in request_text
    assert "studentId" not in request_text
    assert "secret-gms-key" not in request_text

    system_text = captured["input"][0]["content"][0]["text"]
    assert "다음 회기의 지속 관찰 필요성" in system_text
    assert "동일한 난이도에서 추가 확인" in system_text
    assert "evidenceIds 배열에만" in system_text
    assert "text에는 evidenceId" in system_text

    narrative_schema = captured["text"]["format"]["schema"]
    item_schema = narrative_schema["$defs"]["TeacherReportNarrativeItem"]
    assert "evidenceId" in item_schema["properties"]["text"]["description"]
    assert "이 배열에만" in item_schema["properties"]["evidenceIds"]["description"]


def test_gms_narrator_rejects_invalid_model_contract() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "completed", "output_text": '{"unexpected":true}'},
        )

    with pytest.raises(TeacherReportNarratorError):
        _narrator(httpx.MockTransport(handler)).narrate(_evidence())
