from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

NonEmptyText = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ShortIdentifier = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
PolicyHash = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
SkillCode = Annotated[
    StrictStr,
    StringConstraints(
        pattern=(
            r"^(?:"
            r"ONSET_[ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ]|"
            r"NUCLEUS_[ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ]|"
            r"CODA_[ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ]|"
            r"HAS_(?:"
            r"TENSE_ONSET|ASPIRATED_ONSET|GLIDE_VOWEL|COMPOUND_VOWEL|"
            r"BATCHIM|DOUBLE_CODA|COMPLEX_CODA"
            r")|"
            r"PHONO_(?:"
            r"LIAISON|N_INSERTION|PALATALIZATION|NASALIZATION|"
            r"LIQUIDIZATION|TENSIFICATION|ASPIRATION|H_DELETION|"
            r"CODA_NEUTRALIZATION|CLUSTER_SIMPLIFICATION|GLIDE_REDUCTION"
            r")"
            r")$"
        )
    ),
]

SkillRole = Literal["ALLOWED", "EXCLUDED", "LIMITED", "TARGET"]
BranchInputSource = Literal["CHOICE", "TEXT_CONFIRMED", "STT_CONFIRMED"]
AnalysisStatus = Literal["FULL", "UNRELIABLE", "SURFACE_ONLY"]
QualityStatus = Literal[
    "PASS",
    "BEST_EFFORT",
    "ANALYSIS_DEGRADED",
    "REVIEW",
    "REJECTED",
]
SkillQualityStatus = Literal["PASS", "OVER_LIMIT", "OUTSIDE_TARGET", "UNVERIFIED"]


def _to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return "".join((head, *(part.title() for part in tail)))


class StoryPageContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=False,
        extra="forbid",
    )


class IntegerRangePayload(StoryPageContractModel):
    min: StrictInt = Field(ge=0, le=200)
    max: StrictInt = Field(ge=0, le=200)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.min > self.max:
            raise ValueError("min must not exceed max")
        return self


class StoryPageContentContract(StoryPageContractModel):
    sentence_count: Literal[3, 4]
    preferred_written_syllables: IntegerRangePayload
    accepted_written_syllables: IntegerRangePayload
    direct_dialogue_count: StrictInt = Field(ge=0, le=2)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        preferred = self.preferred_written_syllables
        accepted = self.accepted_written_syllables
        if accepted.min > preferred.min or preferred.max > accepted.max:
            raise ValueError("preferredWrittenSyllables must be inside acceptedWrittenSyllables")
        return self


class StoryPageSkillPolicy(StoryPageContractModel):
    code: SkillCode
    role: SkillRole
    max_occurrences: StrictInt | None = Field(default=None, ge=0, le=200)
    target_min: StrictInt | None = Field(default=None, ge=0, le=200)
    target_max: StrictInt | None = Field(default=None, ge=0, le=200)
    unit_penalty: StrictFloat = Field(default=1.0, gt=0, le=20)

    @model_validator(mode="after")
    def validate_role_fields(self) -> Self:
        if self.role in {"EXCLUDED", "LIMITED"}:
            if self.max_occurrences is None:
                raise ValueError(f"maxOccurrences is required for {self.role}")
            if self.target_min is not None or self.target_max is not None:
                raise ValueError("targetMin and targetMax are allowed only for TARGET")
        elif self.role == "TARGET":
            if self.target_min is None or self.target_max is None:
                raise ValueError("targetMin and targetMax are required for TARGET")
            if self.target_min > self.target_max:
                raise ValueError("targetMin must not exceed targetMax")
            if self.max_occurrences is not None:
                raise ValueError("maxOccurrences is not allowed for TARGET")
        elif any(
            value is not None
            for value in (
                self.max_occurrences,
                self.target_min,
                self.target_max,
            )
        ):
            raise ValueError("ALLOWED must not define occurrence constraints")
        return self


class StoryPageGenerationProfile(StoryPageContractModel):
    schema_version: Literal[2]
    generation_profile_version: StrictInt = Field(ge=1)
    source_reading_profile_version: StrictInt = Field(ge=1)
    compiler_version: ShortIdentifier
    policy_hash: PolicyHash
    content_contract: StoryPageContentContract
    skills: list[StoryPageSkillPolicy] = Field(max_length=100)
    protected_terms: list[NonEmptyText] = Field(max_length=50)

    @model_validator(mode="after")
    def validate_unique_values(self) -> Self:
        codes = [skill.code for skill in self.skills]
        if len(codes) != len(set(codes)):
            raise ValueError("skill codes must be unique")
        if len(self.protected_terms) != len(set(self.protected_terms)):
            raise ValueError("protectedTerms must be unique")
        return self


class StoryBeatPayload(StoryPageContractModel):
    beat_id: ShortIdentifier
    goal: NonEmptyText
    question_focus: NonEmptyText | None
    allowed_branch_slots: list[ShortIdentifier] = Field(max_length=20)


class StoryTemplatePayload(StoryPageContractModel):
    template_id: StrictInt = Field(ge=1)
    version: StrictInt = Field(ge=1)
    title: NonEmptyText
    context: NonEmptyText
    current_beat: StoryBeatPayload


class StoryCharacterPayload(StoryPageContractModel):
    character_id: ShortIdentifier
    name: NonEmptyText
    role: NonEmptyText
    immutable_traits: list[NonEmptyText] = Field(max_length=20)


class RecentStoryPagePayload(StoryPageContractModel):
    page_number: StrictInt = Field(ge=1, le=4)
    sentences: list[NonEmptyText] = Field(min_length=1, max_length=5)
    question: NonEmptyText | None


class StoryStatePayload(StoryPageContractModel):
    rolling_summary: StrictStr = Field(max_length=4000)
    resolved_facts: list[NonEmptyText] = Field(max_length=100)
    unresolved_hooks: list[NonEmptyText] = Field(max_length=50)
    recent_pages: list[RecentStoryPagePayload] = Field(max_length=8)
    characters: list[StoryCharacterPayload] = Field(max_length=30)
    last_question: NonEmptyText | None

    @model_validator(mode="after")
    def validate_unique_characters(self) -> Self:
        character_ids = [character.character_id for character in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("storyState characterId values must be unique")
        return self


class StoryPagePlanPayload(StoryPageContractModel):
    page_number: StrictInt = Field(ge=1, le=4)
    locked_event: NonEmptyText
    required_characters: list[ShortIdentifier] = Field(max_length=20)
    required_concepts: list[NonEmptyText] = Field(max_length=20)
    question_focus: NonEmptyText | None


class StoryBranchInputPayload(StoryPageContractModel):
    source: BranchInputSource
    text: NonEmptyText = Field(
        description=("Confirmed, de-identified child input. Raw STT and PII are forbidden.")
    )


class StoryPageGenerateRequest(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[2]
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
    page_plan: StoryPagePlanPayload
    branch_input: StoryBranchInputPayload | None = None
    generation_profile: StoryPageGenerationProfile

    @model_validator(mode="after")
    def validate_character_references(self) -> Self:
        known_characters = {character.character_id for character in self.story_state.characters}
        unknown = set(self.page_plan.required_characters) - known_characters
        if unknown:
            raise ValueError(
                "pagePlan.requiredCharacters contains unknown characterId values: "
                + ", ".join(sorted(unknown))
            )
        expects_branch = self.page_plan.page_number == 4 and not self.conclude
        if expects_branch and self.page_plan.question_focus is None:
            raise ValueError("pagePlan.questionFocus is required for a non-concluding page 4")
        if not expects_branch and self.page_plan.question_focus is not None:
            raise ValueError("pagePlan.questionFocus is allowed only for a non-concluding page 4")
        return self


class GeneratedStoryPagePayload(StoryPageContractModel):
    page_number: StrictInt = Field(ge=1, le=4)
    sentences: list[NonEmptyText] = Field(min_length=4, max_length=4)
    question: NonEmptyText | None
    choices: list[NonEmptyText] = Field(max_length=3)
    requires_branch_input: StrictBool

    @model_validator(mode="after")
    def validate_branch_contract(self) -> Self:
        if self.requires_branch_input:
            if self.question is None or len(self.choices) != 3:
                raise ValueError("a branching page requires one question and exactly three choices")
            if len(set(self.choices)) != 3:
                raise ValueError("branch choices must be distinct")
        elif self.question is not None or self.choices:
            raise ValueError("a non-branching page must not contain a question or choices")
        return self


class StoryPageSkillQuality(StoryPageContractModel):
    code: SkillCode
    role: SkillRole
    status: SkillQualityStatus
    occurrences: StrictInt | None = Field(ge=0)
    max_occurrences: StrictInt | None = Field(ge=0)
    target_min: StrictInt | None = Field(ge=0)
    target_max: StrictInt | None = Field(ge=0)
    overage: StrictInt | None = Field(ge=0)
    target_distance: StrictInt | None = Field(ge=0)
    weighted_risk: StrictFloat = Field(ge=0)


class StoryPageQualityPayload(StoryPageContractModel):
    status: QualityStatus
    analysis_status: AnalysisStatus
    contract_pass: StrictBool
    contract_failures: list[ShortIdentifier] = Field(max_length=20)
    written_syllable_count: StrictInt = Field(ge=0)
    direct_dialogue_count: StrictInt = Field(ge=0)
    excluded_overage_count: StrictInt = Field(ge=0)
    limited_overage_count: StrictInt = Field(ge=0)
    risk_per_10: StrictFloat = Field(ge=0)
    per_skill: list[StoryPageSkillQuality] = Field(max_length=100)


class StoryPageGenerationProvenance(StoryPageContractModel):
    provider: ShortIdentifier
    model: ShortIdentifier
    prompt_version: ShortIdentifier
    generation_profile_version: StrictInt = Field(ge=1)
    policy_hash: PolicyHash
    candidate_count: StrictInt = Field(ge=1, le=10)
    selected_candidate_id: ShortIdentifier
    api_call_count: StrictInt = Field(ge=1, le=20)
    repair_attempted: StrictBool
    repair_accepted: StrictBool
    changed_sentence_numbers: list[StrictInt] = Field(max_length=2)
    repair_decision_reasons: list[ShortIdentifier] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_repair_flags(self) -> Self:
        if self.repair_accepted and not self.repair_attempted:
            raise ValueError("repairAccepted requires repairAttempted")
        if self.changed_sentence_numbers and not self.repair_attempted:
            raise ValueError("changedSentenceNumbers requires repairAttempted")
        if any(
            sentence_number < 1 or sentence_number > 4
            for sentence_number in self.changed_sentence_numbers
        ):
            raise ValueError("changedSentenceNumbers must be between 1 and 4")
        if len(self.changed_sentence_numbers) != len(set(self.changed_sentence_numbers)):
            raise ValueError("changedSentenceNumbers must be unique")
        return self


class StoryPageTimingPayload(StoryPageContractModel):
    generation: StrictFloat = Field(ge=0)
    analysis: StrictFloat = Field(ge=0)
    repair: StrictFloat = Field(ge=0)
    total: StrictFloat = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total < max(
            self.generation,
            self.analysis,
            self.repair,
        ):
            raise ValueError("timingMs.total must cover every measured stage")
        return self


class StoryStatePatchPayload(StoryPageContractModel):
    expected_base_revision: StrictInt = Field(ge=0)
    rolling_summary: StrictStr = Field(max_length=4000)
    resolved_facts_added: list[NonEmptyText] = Field(max_length=20)
    unresolved_hooks_added: list[NonEmptyText] = Field(max_length=20)
    unresolved_hooks_removed: list[NonEmptyText] = Field(max_length=20)
    characters_upserted: list[StoryCharacterPayload] = Field(max_length=10)
    last_question: NonEmptyText | None


class StoryPageGenerateResponse(StoryPageContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[2]
    generation_id: ShortIdentifier
    story_id: StrictInt = Field(ge=1)
    story_revision: StrictInt = Field(ge=0)
    chapter_number: StrictInt = Field(ge=1)
    page: GeneratedStoryPagePayload
    quality: StoryPageQualityPayload
    generation: StoryPageGenerationProvenance
    timing_ms: StoryPageTimingPayload
    state_patch: StoryStatePatchPayload

    def validate_against_request(
        self,
        request: StoryPageGenerateRequest,
    ) -> Self:
        if self.request_id != request.request_id:
            raise ValueError("response requestId must match request requestId")
        if self.story_id != request.story_id:
            raise ValueError("response storyId must match request storyId")
        if self.story_revision != request.story_revision:
            raise ValueError("response storyRevision must match request storyRevision")
        if self.chapter_number != request.chapter_number:
            raise ValueError("response chapterNumber must match request chapterNumber")
        if self.page.page_number != request.page_plan.page_number:
            raise ValueError("response pageNumber must match pagePlan.pageNumber")
        if self.state_patch.expected_base_revision != request.story_revision:
            raise ValueError("statePatch.expectedBaseRevision must match request storyRevision")
        expects_branch = request.page_plan.page_number == 4 and not request.conclude
        if self.page.requires_branch_input != expects_branch:
            raise ValueError("requiresBranchInput is true only for a non-concluding page 4")
        if len(self.page.sentences) != (request.generation_profile.content_contract.sentence_count):
            raise ValueError("response sentence count must match generationProfile.contentContract")
        return self


__all__ = [
    "GeneratedStoryPagePayload",
    "IntegerRangePayload",
    "StoryBeatPayload",
    "StoryBranchInputPayload",
    "StoryCharacterPayload",
    "StoryPageContentContract",
    "StoryPageContractModel",
    "StoryPageGenerateRequest",
    "StoryPageGenerateResponse",
    "StoryPageGenerationProfile",
    "StoryPageGenerationProvenance",
    "StoryPagePlanPayload",
    "StoryPageQualityPayload",
    "StoryPageSkillPolicy",
    "StoryPageSkillQuality",
    "StoryPageTimingPayload",
    "StoryStatePatchPayload",
    "StoryStatePayload",
    "StoryTemplatePayload",
]
