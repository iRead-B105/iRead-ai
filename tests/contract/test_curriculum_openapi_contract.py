from __future__ import annotations

from iread_ai.app import app


def test_curriculum_recommendation_is_published_in_openapi() -> None:
    operation = app.openapi()["paths"]["/api/v1/curricula/recommend"]["post"]

    assert operation["tags"] == ["recommendation"]
    assert operation["requestBody"]["required"] is True
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]
