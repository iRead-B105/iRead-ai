from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from iread_ai.contracts.story_page import (
    NonEmptyText,
    PolicyHash,
    ShortIdentifier,
    StoryBranchInputPayload,
    StoryPageContractModel,
    StoryPageGenerationProfile,
    StoryPageQualityPayload,
    StoryStatePatchPayload,
    StoryStatePayload,
    StoryTemplatePayload,
)

StoryVisualEmotionType = Literal[
    "CALM",
    "HAPPY",
    "EXCITED",
    "CURIOUS",
    "SURPRISED",
    "WORRIED",
    "AFRAID",
    "SAD",
    "DISAPPOINTED",
    "RELIEVED",
    "CONFIDENT",
    "FOCUSED",
    "ANGRY",
]
StoryVisualEmotionIntensity = Literal["LOW", "MEDIUM", "HIGH"]
StoryVisualShot = Literal[
    "WIDE_ESTABLISHING",
    "WIDE_THREE_QUARTER",
    "MEDIUM_TWO_SHOT",
    "MEDIUM_FULL",
    "CLOSE_UP",
]


class StoryVisualEmotionPayload(StoryPageContractModel):
    type: StoryVisualEmotionType
    intensity: StoryVisualEmotionIntensity


class StoryVisualCharacterPayload(StoryPageContractModel):
    character_id: ShortIdentifier
    present: StrictBool
    position: NonEmptyText | None
    orientation: NonEmptyText | None
    gaze_target: NonEmptyText | None
    action: NonEmptyText | None
    emotion: StoryVisualEmotionPayload | None

    @model_validator(mode="after")
    def validate_presence(self) -> Self:
        scene_values = (
            self.position,
            self.orientation,
            self.gaze_target,
            self.action,
            self.emotion,
        )
        if self.present and any(value is None for value in scene_values):
            raise ValueError(
                "a present character requires position, orientation, "
                "gazeTarget, action, and emotion"
            )
        if not self.present and any(value is not None for value in scene_values):
            raise ValueError("an absent character must not define scene details or emotion")
        return self


class StoryVisualScenePayload(StoryPageContractModel):
    shot: StoryVisualShot
    characters: list[StoryVisualCharacterPayload] = Field(
        max_length=30,
    )
    must_include: list[NonEmptyText] = Field(max_length=20)
    must_avoid: list[NonEmptyText] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_scene(self) -> Self:
        character_ids = [character.character_id for character in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("visualScene characterId values must be unique")
        if len(self.must_include) != len(set(self.must_include)):
            raise ValueError("mustInclude values must be unique")
        if len(self.must_avoid) != len(set(self.must_avoid)):
            raise ValueError("mustAvoid values must be unique")
        overlap = set(self.must_include) & set(self.must_avoid)
        if overlap:
            raise ValueError("mustInclude and mustAvoid must not contain the same value")
        return self


class StoryChapterEventPayload(StoryPageContractModel):
    event_id: ShortIdentifier
    locked_event: NonEmptyText
    required_characters: list[ShortIdentifier] = Field(max_length=20)
    required_concepts: list[NonEmptyText] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        if len(self.required_characters) != len(set(self.required_characters)):
            raise ValueError("requiredCharacters must be unique")
        if len(self.required_concepts) != len(set(self.required_concepts)):
            raise ValueError("requiredConcepts must be unique")
        return self


class StoryChapterPlanPayload(StoryPageContractModel):
    ordered_events: list[StoryChapterEventPayload] = Field(
        min_length=1,
        max_length=4,
    )
    min_pages: StrictInt = Field(ge=2, le=4)
    max_pages: StrictInt = Field(ge=2, le=4)
    question_focus: NonEmptyText | None

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.min_pages > self.max_pages:
            raise ValueError("minPages must not exceed maxPages")
        event_ids = [event.event_id for event in self.ordered_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("orderedEvents eventId values must be unique")
        return self


class StoryChapterGenerateRequest(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[3]
    story_id: StrictInt = Field(ge=1)
    student_id: StrictInt = Field(
        ge=1,
        description=("Backend routing identifier. It must never be forwarded to the model."),
    )
    story_revision: StrictInt = Field(ge=0)
    chapter_number: StrictInt = Field(ge=1)
    conclude: StrictBool
    story_template: StoryTemplatePayload
    story_state: StoryStatePayload
    chapter_plan: StoryChapterPlanPayload
    branch_input: StoryBranchInputPayload | None = None
    generation_profile: StoryPageGenerationProfile

    @model_validator(mode="after")
    def validate_request_invariants(self) -> Self:
        known_characters = {character.character_id for character in self.story_state.characters}
        required_characters = {
            character_id
            for event in self.chapter_plan.ordered_events
            for character_id in event.required_characters
        }
        unknown = required_characters - known_characters
        if unknown:
            raise ValueError(
                "chapterPlan.orderedEvents contains unknown characterId values: "
                + ", ".join(sorted(unknown))
            )

        if self.conclude and self.chapter_plan.question_focus is not None:
            raise ValueError("chapterPlan.questionFocus is not allowed for a concluding chapter")
        if not self.conclude and self.chapter_plan.question_focus is None:
            raise ValueError("chapterPlan.questionFocus is required for a non-concluding chapter")
        return self


class GeneratedStoryChapterPagePayload(StoryPageContractModel):
    page_number: StrictInt = Field(ge=1, le=4)
    sentences: list[NonEmptyText] = Field(min_length=3, max_length=4)
    visual_scene: StoryVisualScenePayload
    question: NonEmptyText | None
    subtitle: NonEmptyText | None
    choices: list[NonEmptyText] = Field(max_length=3)
    requires_branch_input: StrictBool

    @model_validator(mode="after")
    def validate_branch_contract(self) -> Self:
        if self.requires_branch_input:
            if self.question is None or self.subtitle is None or len(self.choices) != 3:
                raise ValueError(
                    "a branching page requires a subtitle, one question, and exactly three choices"
                )
            if len(set(self.choices)) != 3:
                raise ValueError("branch choices must be distinct")
        elif self.question is not None or self.subtitle is not None or self.choices:
            raise ValueError(
                "a non-branching page must not contain a subtitle, question, or choices"
            )
        return self


class StoryChapterPageQualityPayload(StoryPageContractModel):
    page_number: StrictInt = Field(ge=1, le=4)
    quality: StoryPageQualityPayload


class StoryChapterQualityPayload(StoryPageContractModel):
    chapter: StoryPageQualityPayload
    pages: list[StoryChapterPageQualityPayload] = Field(
        min_length=2,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_quality_totals(self) -> Self:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("quality pages must be ordered from 1 through page count")

        totals = {
            "written_syllable_count": sum(
                page.quality.written_syllable_count for page in self.pages
            ),
            "direct_dialogue_count": sum(page.quality.direct_dialogue_count for page in self.pages),
            "excluded_overage_count": sum(
                page.quality.excluded_overage_count for page in self.pages
            ),
            "limited_overage_count": sum(page.quality.limited_overage_count for page in self.pages),
        }
        for field_name, expected in totals.items():
            if getattr(self.chapter, field_name) != expected:
                alias = "".join(
                    (
                        field_name.split("_")[0],
                        *(part.title() for part in field_name.split("_")[1:]),
                    )
                )
                raise ValueError(f"chapter {alias} must equal the sum of page qualities")
        return self


class StoryChapterChangedSentencePayload(StoryPageContractModel):
    global_sentence_number: StrictInt = Field(ge=1, le=16)
    page_number: StrictInt = Field(ge=1, le=4)
    sentence_number: StrictInt = Field(ge=1, le=4)


class StoryChapterGenerationProvenance(StoryPageContractModel):
    provider: ShortIdentifier
    model: ShortIdentifier
    prompt_version: ShortIdentifier
    generation_profile_version: StrictInt = Field(ge=1)
    policy_hash: PolicyHash
    candidate_count: StrictInt = Field(ge=1, le=10)
    selected_candidate_id: ShortIdentifier
    page_count: StrictInt = Field(ge=2, le=4)
    api_call_count: StrictInt = Field(ge=1, le=20)
    repair_attempted: StrictBool
    repair_accepted: StrictBool
    changed_sentences: list[StoryChapterChangedSentencePayload] = Field(
        max_length=2,
    )
    repair_decision_reasons: list[ShortIdentifier] = Field(max_length=20)
    visual_scene_status: Literal[
        "LLM_GENERATED",
        "DETERMINISTIC_FALLBACK",
        "MOCK",
    ]
    visual_scene_model: ShortIdentifier
    visual_scene_prompt_version: ShortIdentifier
    visual_scene_fallback_reason: ShortIdentifier | None

    @model_validator(mode="after")
    def validate_repair_fields(self) -> Self:
        if self.repair_accepted and not self.repair_attempted:
            raise ValueError("repairAccepted requires repairAttempted")
        if self.changed_sentences and not self.repair_attempted:
            raise ValueError("changedSentences requires repairAttempted")
        if (
            self.visual_scene_status == "DETERMINISTIC_FALLBACK"
            and self.visual_scene_fallback_reason is None
        ):
            raise ValueError("visualSceneFallbackReason is required for fallback")
        if (
            self.visual_scene_status != "DETERMINISTIC_FALLBACK"
            and self.visual_scene_fallback_reason is not None
        ):
            raise ValueError("visualSceneFallbackReason is allowed only for fallback")

        global_numbers = [sentence.global_sentence_number for sentence in self.changed_sentences]
        positions = [
            (sentence.page_number, sentence.sentence_number) for sentence in self.changed_sentences
        ]
        if len(global_numbers) != len(set(global_numbers)):
            raise ValueError("changedSentences globalSentenceNumber values must be unique")
        if len(positions) != len(set(positions)):
            raise ValueError("changedSentences page and sentence positions must be unique")
        return self


class StoryChapterTimingPayload(StoryPageContractModel):
    generation: StrictFloat = Field(ge=0)
    analysis: StrictFloat = Field(ge=0)
    pagination: StrictFloat = Field(ge=0)
    repair: StrictFloat = Field(ge=0)
    visual_scene: StrictFloat = Field(ge=0)
    total: StrictFloat = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total < max(
            self.generation,
            self.analysis,
            self.pagination,
            self.repair,
            self.visual_scene,
        ):
            raise ValueError("timingMs.total must cover every measured stage")
        return self


class StoryChapterGenerateResponse(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[3]
    generation_id: ShortIdentifier
    story_id: StrictInt = Field(ge=1)
    story_revision: StrictInt = Field(ge=0)
    chapter_number: StrictInt = Field(ge=1)
    pages: list[GeneratedStoryChapterPagePayload] = Field(
        min_length=2,
        max_length=4,
    )
    quality: StoryChapterQualityPayload
    generation: StoryChapterGenerationProvenance
    timing_ms: StoryChapterTimingPayload
    state_patch: StoryStatePatchPayload

    @model_validator(mode="after")
    def validate_response_invariants(self) -> Self:
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("pages must be ordered from 1 through page count")
        if len(self.quality.pages) != len(self.pages):
            raise ValueError("quality.pages must contain one entry for every page")
        if [page_quality.page_number for page_quality in self.quality.pages] != page_numbers:
            raise ValueError("quality.pages must match generated page numbers")
        if self.generation.page_count != len(self.pages):
            raise ValueError("generation.pageCount must equal generated page count")

        for page in self.pages[:-1]:
            if page.requires_branch_input:
                raise ValueError("requiresBranchInput may be true only on the final page")

        for changed in self.generation.changed_sentences:
            if changed.page_number > len(self.pages):
                raise ValueError("changedSentences references an unknown page")
            page = self.pages[changed.page_number - 1]
            if changed.sentence_number > len(page.sentences):
                raise ValueError("changedSentences references an unknown sentence")
            expected_global_number = (
                sum(len(previous.sentences) for previous in self.pages[: changed.page_number - 1])
                + changed.sentence_number
            )
            if changed.global_sentence_number != expected_global_number:
                raise ValueError(
                    "changedSentences globalSentenceNumber does not match its page position"
                )
        return self

    def validate_against_request(
        self,
        request: StoryChapterGenerateRequest,
    ) -> Self:
        if self.request_id != request.request_id:
            raise ValueError("response requestId must match request requestId")
        if self.story_id != request.story_id:
            raise ValueError("response storyId must match request storyId")
        if self.story_revision != request.story_revision:
            raise ValueError("response storyRevision must match request storyRevision")
        if self.chapter_number != request.chapter_number:
            raise ValueError("response chapterNumber must match request chapterNumber")
        if self.state_patch.expected_base_revision != request.story_revision:
            raise ValueError("statePatch.expectedBaseRevision must match request storyRevision")

        page_count = len(self.pages)
        if not (request.chapter_plan.min_pages <= page_count <= request.chapter_plan.max_pages):
            raise ValueError("response page count must be inside chapterPlan page bounds")

        final_page = self.pages[-1]
        expects_branch = not request.conclude
        if final_page.requires_branch_input != expects_branch:
            raise ValueError(
                "only the final page of a non-concluding chapter may require branch input"
            )
        if self.generation.generation_profile_version != (
            request.generation_profile.generation_profile_version
        ):
            raise ValueError("generationProfileVersion must match the request snapshot")
        if self.generation.policy_hash != request.generation_profile.policy_hash:
            raise ValueError("generation policyHash must match the request snapshot")
        expected_character_ids = [
            character.character_id for character in request.story_state.characters
        ]
        for page in self.pages:
            actual_character_ids = [
                character.character_id for character in page.visual_scene.characters
            ]
            if actual_character_ids != expected_character_ids:
                raise ValueError(
                    "visualScene characters must contain every storyState "
                    "character exactly once and in request order"
                )
        return self


__all__ = [
    "GeneratedStoryChapterPagePayload",
    "StoryChapterChangedSentencePayload",
    "StoryChapterEventPayload",
    "StoryChapterGenerateRequest",
    "StoryChapterGenerateResponse",
    "StoryChapterGenerationProvenance",
    "StoryChapterPageQualityPayload",
    "StoryChapterPlanPayload",
    "StoryChapterQualityPayload",
    "StoryChapterTimingPayload",
    "StoryVisualCharacterPayload",
    "StoryVisualEmotionIntensity",
    "StoryVisualEmotionPayload",
    "StoryVisualEmotionType",
    "StoryVisualScenePayload",
    "StoryVisualShot",
]
