from __future__ import annotations

from iread_ai.config import Settings
from iread_ai.main import create_app

PATH = "/api/v1/story/images/generate"


def _openapi() -> dict[str, object]:
    app = create_app(
        settings=Settings(
            app_env="test",
            internal_api_key="story-image-contract-key",
            story_provider="mock",
            gms_key=None,
        )
    )
    return app.openapi()


def test_story_image_operation_uses_auth_idempotency_and_canonical_models() -> None:
    schema = _openapi()
    operation = schema["paths"][PATH]["post"]

    assert operation["operationId"] == "generateStoryPageImage"
    assert operation["x-idempotency-required"] is True
    assert operation["x-timeout-ms"] == 190000
    assert {"apiKeyAuth": []} in operation["security"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StoryImageGenerateRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/StoryImageGenerateResponse"}


def test_story_image_request_accepts_visual_scene_and_id_only_references() -> None:
    schemas = _openapi()["components"]["schemas"]
    request = schemas["StoryImageGenerateRequest"]

    assert request["additionalProperties"] is False
    assert set(request["required"]) == {
        "requestId",
        "schemaVersion",
        "storyId",
        "storyRevision",
        "chapterNumber",
        "pageNumber",
        "sentences",
        "visualScene",
        "storyContext",
    }
    assert "characterReferences" in request["properties"]
    assert "previousImageBase64" not in request["properties"]
    assert "referenceImagePath" not in request["properties"]
    assert request["properties"]["visualScene"] == {
        "$ref": "#/components/schemas/StoryVisualScenePayload"
    }
    reference = schemas["StoryImageCharacterReferencePayload"]
    assert reference["additionalProperties"] is False
    assert set(reference["required"]) == {"characterId"}
    assert set(reference["properties"]) == {"characterId"}


def test_story_image_response_returns_image_and_generation_metadata() -> None:
    schemas = _openapi()["components"]["schemas"]
    response = schemas["StoryImageGenerateResponse"]

    assert response["additionalProperties"] is False
    assert set(response["required"]) == {
        "requestId",
        "schemaVersion",
        "imageId",
        "mimeType",
        "imageBase64",
        "model",
        "promptVersion",
        "timingMs",
    }
    assert set(response["properties"]["mimeType"]["enum"]) == {
        "image/png",
        "image/jpeg",
        "image/webp",
    }
