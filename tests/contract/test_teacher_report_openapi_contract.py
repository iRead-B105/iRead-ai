from __future__ import annotations

from iread_ai.config import Settings
from iread_ai.main import create_app

PATH = "/api/v1/reports/analyze"


def _openapi() -> dict[str, object]:
    app = create_app(
        settings=Settings(
            app_env="test",
            internal_api_key="teacher-report-contract-key",
            story_provider="mock",
            generation_provider="mock",
            gms_key=None,
        )
    )
    return app.openapi()


def test_teacher_report_operation_uses_auth_idempotency_and_canonical_models() -> None:
    schema = _openapi()
    operation = schema["paths"][PATH]["post"]

    assert operation["operationId"] == "analyzeTeacherReport"
    assert operation["x-idempotency-required"] is True
    assert operation["x-timeout-ms"] == 30000
    assert {"apiKeyAuth": []} in operation["security"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TeacherReportAnalyzeRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/TeacherReportAnalyzeResponse"
    }


def test_teacher_report_request_is_aggregate_only_and_strict() -> None:
    schemas = _openapi()["components"]["schemas"]
    request = schemas["TeacherReportAnalyzeRequest"]
    feature = schemas["TeacherReportFeatureProfile"]

    assert request["additionalProperties"] is False
    assert set(request["required"]) == {
        "requestId",
        "schemaVersion",
        "profileAnalysisVersion",
        "featureProfiles",
        "gazeTrend",
    }
    assert "studentId" not in request["properties"]
    assert "rawGazeData" not in request["properties"]
    assert feature["additionalProperties"] is False
    assert {
        "accuracyRate",
        "weaknessScore",
        "confidence",
        "evidenceCount",
        "avgFixationDurationMs",
        "avgRegressionCount",
        "skipRate",
    }.issubset(feature["properties"])
    assert feature["properties"]["weaknessScore"]["maximum"] == 1
    assert feature["properties"]["weaknessScore"]["minimum"] == 0
    pronunciation = feature["properties"]["avgPronunciationScore"]["anyOf"][0]
    assert pronunciation["maximum"] == 100
    assert pronunciation["minimum"] == 0


def test_teacher_report_response_maps_to_existing_snapshot_fields() -> None:
    response = _openapi()["components"]["schemas"]["TeacherReportAnalyzeResponse"]

    assert response["additionalProperties"] is False
    assert set(response["required"]) == {
        "requestId",
        "schemaVersion",
        "analysisVersion",
        "summaryProvider",
        "dataSufficiency",
        "improvedPatterns",
        "persistentDifficultyPatterns",
        "gazeDescriptions",
    }
