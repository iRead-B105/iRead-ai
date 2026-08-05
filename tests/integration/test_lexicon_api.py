from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from iread_ai.config import Settings
from iread_ai.lexicon.service import LexiconPaletteService
from iread_ai.main import create_app
from tests.unit.test_lexicon_service import _create_database


def test_lexicon_status_and_palette_api(tmp_path: Path) -> None:
    database = tmp_path / "lexicon.sqlite3"
    _create_database(database)
    settings = Settings(
        app_env="test",
        internal_api_key="test-key",
        story_provider="mock",
        generation_provider="mock",
        lexicon_database_path=database,
    )
    app = create_app(
        settings=settings,
        lexicon_service=LexiconPaletteService(database),
    )

    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/lexicon/status")
        status = client.get(
            "/api/v1/lexicon/status",
            headers={"X-API-Key": "test-key"},
        )
        palette = client.post(
            "/api/v1/lexicon/palettes/query",
            headers={"X-API-Key": "test-key"},
            json={
                "requestId": "api-palette-1",
                "schemaVersion": 1,
                "targetFeatures": [
                    {
                        "featureCode": "GRAPHEME.ONSET.TENSE.ㄲ",
                        "weaknessScore": 0.8,
                        "confidence": 0.9,
                    }
                ],
                "excludedFeatures": ["PHONO_LIAISON"],
                "requireTarget": True,
                "limit": 10,
            },
        )

    assert unauthorized.status_code == 401
    assert status.status_code == 200
    assert status.json()["status"] == "READY"
    assert palette.status_code == 200
    assert [item["surface"] for item in palette.json()["items"]] == ["꿀"]
