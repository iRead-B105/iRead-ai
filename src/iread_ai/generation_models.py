"""훈련·이야기·이미지 생성 API 계약 모델."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainingTargetFeature(ContractModel):
    featureCode: str = Field(min_length=1)
    weaknessScore: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidenceCount: int = Field(ge=0)


class TrainingCandidateRequest(ContractModel):
    requestId: str = Field(min_length=1, max_length=128)
    schemaVersion: int = Field(ge=1)
    trainingType: str = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=9)
    difficulty: int = Field(ge=1, le=5)
    targetFeatures: list[TrainingTargetFeature] = Field(default_factory=list, max_length=2)
    excludedFeatures: list[str] = Field(default_factory=list)
    additionalPrompt: str = ""
    outputTemplate: dict[str, Any]
    useLexicon: bool = True
    recommendedWords: list[str] = Field(default_factory=list, max_length=40)
    recommendedWordsByFeature: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_feature_policy(self) -> "TrainingCandidateRequest":
        excluded = [feature.strip() for feature in self.excludedFeatures]
        if any(not feature for feature in excluded) or len(excluded) != len(set(excluded)):
            raise ValueError("excludedFeatures must be unique and non-empty")
        target = {feature.featureCode for feature in self.targetFeatures}
        overlap = target.intersection(excluded)
        if overlap:
            raise ValueError("targetFeatures and excludedFeatures must not overlap")
        for feature_code, words in self.recommendedWordsByFeature.items():
            if not feature_code.strip():
                raise ValueError("recommendedWordsByFeature keys must be non-empty")
            if not words or len(words) > 20:
                raise ValueError(
                    "recommendedWordsByFeature values must contain between 1 and 20 words"
                )
            normalized = [word.strip() for word in words]
            if any(not word for word in normalized) or len(normalized) != len(set(normalized)):
                raise ValueError(
                    "recommendedWordsByFeature values must contain unique non-empty words"
                )
        return self


TrainingLexicalPolicy = Literal["PSEUDOWORD_ALLOWED", "REAL_WORD_ONLY"]
TrainingGenerationStrategy = Literal[
    "RULE_DB",
    "LLM_WITH_LOCAL_VALIDATION",
    "CURATED_FALLBACK",
]


class TrainingGenerationMetadata(ContractModel):
    provider: str
    model: str
    strategy: TrainingGenerationStrategy
    lexicalPolicy: TrainingLexicalPolicy
    lexiconApplied: bool


class TrainingCandidateResponse(ContractModel):
    type: str
    data: list[dict[str, Any]] = Field(min_length=5, max_length=5)
    generationMetadata: TrainingGenerationMetadata | None = None


TrainingCurriculumArea = Literal[
    "AUTO",
    "LETTER_SOUND",
    "BLENDING",
    "WORD_READING",
    "SENTENCE",
    "FLUENCY",
]


class TrainingActivityRequest(ContractModel):
    requestId: str = Field(min_length=1, max_length=128)
    schemaVersion: int = Field(default=1, ge=1)
    sequence: int = Field(default=1, ge=1, le=5)
    trainingType: str = Field(min_length=1)
    difficulty: int = Field(ge=1, le=5)
    targetFeatures: list[TrainingTargetFeature] = Field(default_factory=list, max_length=2)
    excludedFeatures: list[str] = Field(default_factory=list)
    additionalPrompt: str = ""
    useLexicon: bool = True

    @model_validator(mode="after")
    def validate_feature_policy(self) -> "TrainingActivityRequest":
        _validate_training_feature_policy(self.targetFeatures, self.excludedFeatures)
        return self


class GeneratedTrainingActivity(ContractModel):
    sequence: int = Field(ge=1, le=5)
    templateId: int = Field(ge=1)
    trainingType: str
    name: str
    group: str
    strategy: str
    provider: str
    targetFeatureCodes: list[str]
    rationale: str
    item: dict[str, Any]
    personalization: "TrainingPersonalizationEvidence"


class TrainingCandidateFit(ContractModel):
    candidateIndex: int = Field(ge=0, le=4)
    score: float
    targetOccurrences: dict[str, int]
    excludedOccurrences: dict[str, int]
    paletteWordUses: list[str]
    analysisStatus: str
    analysisError: str | None = None
    writtenSyllableCount: int = Field(default=0, ge=0)
    sentenceSyllableCounts: list[int] = Field(default_factory=list)
    lengthStatus: str = "NOT_APPLICABLE"
    lengthAdjustment: float = 0
    targetLoadStatus: str = "NOT_APPLICABLE"
    targetOccurrenceTotal: int = Field(default=0, ge=0)


class TrainingPersonalizationEvidence(ContractModel):
    lexiconApplied: bool
    recommendedWords: list[str]
    selectedCandidateIndex: int = Field(ge=0, le=4)
    candidates: list[TrainingCandidateFit] = Field(min_length=1, max_length=5)
    generationAttempts: int = Field(default=1, ge=1, le=2)


class TrainingActivityResponse(ContractModel):
    requestId: str
    schemaVersion: int
    activity: GeneratedTrainingActivity


class TrainingSetRequest(ContractModel):
    requestId: str = Field(min_length=1, max_length=128)
    schemaVersion: int = Field(default=1, ge=1)
    curriculumArea: TrainingCurriculumArea = "AUTO"
    activityCount: int = Field(default=5, ge=5, le=5)
    difficulty: int = Field(ge=1, le=5)
    targetFeatures: list[TrainingTargetFeature] = Field(default_factory=list, max_length=2)
    excludedFeatures: list[str] = Field(default_factory=list)
    preferredTrainingTypes: list[str] = Field(default_factory=list, max_length=5)
    additionalPrompt: str = ""
    useLexicon: bool = True

    @model_validator(mode="after")
    def validate_policy(self) -> "TrainingSetRequest":
        _validate_training_feature_policy(self.targetFeatures, self.excludedFeatures)
        if len(self.preferredTrainingTypes) != len(set(self.preferredTrainingTypes)):
            raise ValueError("preferredTrainingTypes must not contain duplicates")
        return self


class TrainingSetResponse(ContractModel):
    requestId: str
    schemaVersion: int
    curriculumArea: TrainingCurriculumArea
    focusFeatureCodes: list[str]
    activities: list[GeneratedTrainingActivity] = Field(min_length=5, max_length=5)


def _validate_training_feature_policy(
    target_features: list[TrainingTargetFeature],
    excluded_features: list[str],
) -> None:
    excluded = [feature.strip() for feature in excluded_features]
    if any(not feature for feature in excluded) or len(excluded) != len(set(excluded)):
        raise ValueError("excludedFeatures must be unique and non-empty")
    target = {feature.featureCode for feature in target_features}
    overlap = target.intersection(excluded)
    if overlap:
        raise ValueError("targetFeatures and excludedFeatures must not overlap")


class GenerateTrainingRequest(ContractModel):
    requestId: str = Field(min_length=1)
    trainingId: int = Field(ge=1)
    studentId: int = Field(ge=1)
    trainingTemplateId: int = Field(ge=1)
    schemaVersion: int = Field(ge=1)
    inputData: dict[str, Any]


class GenerateTrainingResponse(ContractModel):
    requestId: str
    schemaVersion: int
    generatedData: dict[str, Any]


class StoryTemplateData(ContractModel):
    storyTemplateId: int = Field(ge=1)
    title: str = Field(min_length=1)
    context: str = ""


class StoryHistoryLine(ContractModel):
    storyLineId: int = Field(ge=1)
    content: str
    requiresBranchInput: bool


class GenerateStoryRequest(ContractModel):
    requestId: str = Field(min_length=1)
    storyId: int = Field(ge=1)
    studentId: int = Field(ge=1)
    schemaVersion: int = Field(ge=1)
    currentProgress: int = Field(ge=0, le=100)
    storyTemplate: StoryTemplateData


class ContinueStoryRequest(GenerateStoryRequest):
    currentStoryLineId: int = Field(ge=1)
    branchIntent: str = Field(min_length=1)
    history: list[StoryHistoryLine] = Field(default_factory=list)


class StoryBranchOption(ContractModel):
    optionNo: int = Field(ge=1, le=3)
    label: str = Field(min_length=1, max_length=80)


class StoryBranchPrompt(ContractModel):
    subtitle: str = Field(min_length=1, max_length=40)
    options: list[StoryBranchOption] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_options(self) -> "StoryBranchPrompt":
        if self.subtitle != self.subtitle.strip():
            raise ValueError("branch subtitle must not have surrounding whitespace")
        option_numbers = {option.optionNo for option in self.options}
        labels = {option.label.strip() for option in self.options}
        if option_numbers != {1, 2, 3}:
            raise ValueError("branch option numbers must be 1, 2, and 3")
        if len(labels) != 3 or any(not label for label in labels):
            raise ValueError("branch option labels must be unique and non-empty")
        if any(option.label != option.label.strip() for option in self.options):
            raise ValueError("branch option labels must not have surrounding whitespace")
        return self


class GeneratedStoryLine(ContractModel):
    content: str = Field(min_length=1)
    requiresBranchInput: bool
    branchPrompt: StoryBranchPrompt | None = None

    @model_validator(mode="after")
    def validate_branch_prompt(self) -> "GeneratedStoryLine":
        if self.requiresBranchInput != (self.branchPrompt is not None):
            raise ValueError("branchPrompt is required only when requiresBranchInput is true")
        return self


class GenerateStoryResponse(ContractModel):
    requestId: str
    schemaVersion: int
    nextProgress: int = Field(ge=0, le=100)
    completed: bool
    lines: list[GeneratedStoryLine] = Field(min_length=1, max_length=5)


class GenerateImageRequest(ContractModel):
    requestId: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    storyTemplateId: int | None = Field(default=None, ge=1)


class GenerateImageResponse(ContractModel):
    requestId: str
    imageUrl: str
    provider: str


class EvaluateTrainingRequest(ContractModel):
    requestId: str = Field(min_length=1)
    trainingId: int = Field(ge=1)
    studentId: int = Field(ge=1)
    trainingTemplateId: int = Field(ge=1)
    schemaVersion: int = Field(ge=1)
    result: dict[str, Any]


class EvaluateTrainingResponse(ContractModel):
    requestId: str
    schemaVersion: int
    accuracy: float = Field(ge=0, le=100)


class SpeechTranscriptionResponse(ContractModel):
    requestId: str
    transcript: str
    confidence: float = Field(ge=0, le=1)
    durationMs: int = Field(ge=0)


class SpeechSynthesisRequest(ContractModel):
    requestId: str = Field(min_length=1)
    text: str = Field(min_length=1)
    voice: str | None = None
    tempo: float = Field(default=1.0, ge=0.5, le=2.0)
