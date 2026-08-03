from __future__ import annotations

from iread_ai.app import app


def test_deployed_entrypoint_exposes_only_supported_story_surfaces() -> None:
    paths = set(app.openapi()["paths"])

    assert {
        "/health",
        "/api/v1/trainings/candidates",
        "/api/v1/trainings/generate",
        "/api/v1/story/generate",
        "/api/v1/story/continue",
        "/api/v1/images/generate",
        "/api/v1/images/mock/generated.png",
        "/api/v1/speech/pronunciation/analyze",
        "/api/v1/training-sets/generate",
        "/api/v1/training-activities/generate",
        "/api/v1/curricula/recommend",
        "/api/v1/lexicon/status",
        "/api/v1/lexicon/palettes/query",
        "/api/v1/reports/analyze",
        "/api/v3/story/chapters/generate",
        "/api/v1/story/images/generate",
        "/api/dev/story/displayed-chapter-comparison",
    }.issubset(paths)
    assert "/api/v2/story/pages/generate" not in paths
    assert "/api/dev/story/page-comparison" not in paths
    assert "/api/dev/story/displayed-page-comparison" not in paths
