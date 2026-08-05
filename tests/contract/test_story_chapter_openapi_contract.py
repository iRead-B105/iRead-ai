from __future__ import annotations

from fastapi import FastAPI

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)

PATH = "/api/v3/story/chapters/generate"


def _contract_app() -> FastAPI:
    app = FastAPI()

    @app.post(
        PATH,
        operation_id="generatePersonalizedStoryChapter",
        response_model=StoryChapterGenerateResponse,
    )
    async def generate_chapter(
        payload: StoryChapterGenerateRequest,
    ) -> StoryChapterGenerateResponse:
        raise NotImplementedError(payload.request_id)

    return app


def test_story_chapter_v3_openapi_uses_canonical_path_and_models() -> None:
    schema = _contract_app().openapi()
    operation = schema["paths"][PATH]["post"]

    assert operation["operationId"] == "generatePersonalizedStoryChapter"
    assert operation["requestBody"]["required"] is True
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StoryChapterGenerateRequest"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/StoryChapterGenerateResponse"
    }


def test_story_chapter_v3_request_schema_is_strict_and_reuses_profile_v2() -> None:
    schemas = _contract_app().openapi()["components"]["schemas"]
    request_schema = schemas["StoryChapterGenerateRequest"]

    assert request_schema["additionalProperties"] is False
    assert set(request_schema["required"]) == {
        "requestId",
        "schemaVersion",
        "storyId",
        "studentId",
        "storyRevision",
        "chapterNumber",
        "conclude",
        "storyTemplate",
        "storyState",
        "chapterPlan",
        "generationProfile",
    }
    assert request_schema["properties"]["schemaVersion"]["const"] == 3
    assert "branchInput" in request_schema["properties"]
    assert "branchInput" not in request_schema["required"]

    profile_schema = schemas["StoryPageGenerationProfile"]
    assert profile_schema["properties"]["schemaVersion"]["const"] == 2
    assert profile_schema["additionalProperties"] is False


def test_story_chapter_plan_has_ordered_events_and_dynamic_page_bounds() -> None:
    schemas = _contract_app().openapi()["components"]["schemas"]
    plan_schema = schemas["StoryChapterPlanPayload"]
    events_schema = plan_schema["properties"]["orderedEvents"]

    assert plan_schema["additionalProperties"] is False
    assert set(plan_schema["required"]) == {
        "orderedEvents",
        "minPages",
        "maxPages",
        "questionFocus",
    }
    assert events_schema["minItems"] == 1
    assert events_schema["maxItems"] == 4
    assert plan_schema["properties"]["minPages"]["minimum"] == 2
    assert plan_schema["properties"]["minPages"]["maximum"] == 4
    assert plan_schema["properties"]["maxPages"]["minimum"] == 2
    assert plan_schema["properties"]["maxPages"]["maximum"] == 4

    event_schema = schemas["StoryChapterEventPayload"]
    assert set(event_schema["required"]) == {
        "eventId",
        "lockedEvent",
        "requiredCharacters",
        "requiredConcepts",
    }


def test_story_chapter_v3_response_exposes_dynamic_pages_and_quality() -> None:
    schemas = _contract_app().openapi()["components"]["schemas"]
    response_schema = schemas["StoryChapterGenerateResponse"]

    assert response_schema["additionalProperties"] is False
    assert set(response_schema["required"]) == {
        "requestId",
        "schemaVersion",
        "generationId",
        "storyId",
        "storyRevision",
        "chapterNumber",
        "pages",
        "quality",
        "generation",
        "timingMs",
        "statePatch",
    }
    assert response_schema["properties"]["schemaVersion"]["const"] == 3
    assert response_schema["properties"]["pages"]["minItems"] == 2
    assert response_schema["properties"]["pages"]["maxItems"] == 4

    page_schema = schemas["GeneratedStoryChapterPagePayload"]
    sentences_schema = page_schema["properties"]["sentences"]
    assert sentences_schema["minItems"] == 3
    assert sentences_schema["maxItems"] == 4
    choices_schema = page_schema["properties"]["choices"]
    assert choices_schema["maxItems"] == 3
    assert {
        "pageNumber",
        "sentences",
        "visualScene",
        "question",
        "subtitle",
        "choices",
        "requiresBranchInput",
    } == set(page_schema["required"])

    scene_schema = schemas["StoryVisualScenePayload"]
    assert set(scene_schema["required"]) == {
        "shot",
        "characters",
        "mustInclude",
        "mustAvoid",
    }
    character_schema = schemas["StoryVisualCharacterPayload"]
    assert set(character_schema["required"]) == {
        "characterId",
        "present",
        "position",
        "orientation",
        "gazeTarget",
        "action",
        "emotion",
    }
    emotion_schema = schemas["StoryVisualEmotionPayload"]
    assert set(emotion_schema["required"]) == {"type", "intensity"}
    assert set(emotion_schema["properties"]["intensity"]["enum"]) == {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    quality_schema = schemas["StoryChapterQualityPayload"]
    assert set(quality_schema["required"]) == {"chapter", "pages"}
    assert quality_schema["properties"]["pages"]["minItems"] == 2
    assert quality_schema["properties"]["pages"]["maxItems"] == 4


def test_story_chapter_v3_provenance_and_timing_cover_chapter_stages() -> None:
    schemas = _contract_app().openapi()["components"]["schemas"]
    provenance = schemas["StoryChapterGenerationProvenance"]
    timing = schemas["StoryChapterTimingPayload"]

    assert {
        "pageCount",
        "candidateCount",
        "selectedCandidateId",
        "changedSentences",
        "repairDecisionReasons",
        "visualSceneStatus",
        "visualSceneModel",
        "visualScenePromptVersion",
        "visualSceneFallbackReason",
    }.issubset(provenance["properties"])
    assert provenance["properties"]["pageCount"]["minimum"] == 2
    assert provenance["properties"]["pageCount"]["maximum"] == 4
    assert set(timing["required"]) == {
        "generation",
        "analysis",
        "pagination",
        "repair",
        "visualScene",
        "total",
    }


def test_story_chapter_v3_schema_does_not_offer_pii_fields() -> None:
    schemas = _contract_app().openapi()["components"]["schemas"]
    forbidden_properties = {
        "childName",
        "studentName",
        "birthDate",
        "phoneNumber",
        "email",
        "schoolName",
        "guardianName",
    }

    for name in (
        "StoryChapterGenerateRequest",
        "StoryChapterPlanPayload",
        "StoryChapterEventPayload",
        "StoryStatePayload",
        "StoryCharacterPayload",
        "StoryBranchInputPayload",
        "StoryPageGenerationProfile",
    ):
        properties = set(schemas[name].get("properties", {}))
        assert not properties & forbidden_properties
