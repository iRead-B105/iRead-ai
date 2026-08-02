"""훈련·이야기·이미지 생성 API 계약 모델."""

from typing import Any

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
    count: int = Field(default=5, ge=1, le=20)
    difficulty: int = Field(ge=1, le=5)
    targetFeatures: list[TrainingTargetFeature] = Field(default_factory=list, max_length=2)
    excludedFeatures: list[str] = Field(default_factory=list)
    additionalPrompt: str = ""
    outputTemplate: dict[str, Any]


class TrainingCandidateResponse(ContractModel):
    type: str
    data: list[dict[str, Any]]


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
    options: list[StoryBranchOption] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_options(self) -> "StoryBranchPrompt":
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
            raise ValueError(
                "branchPrompt is required only when requiresBranchInput is true"
            )
        return self


class GenerateStoryResponse(ContractModel):
    requestId: str
    schemaVersion: int
    nextProgress: int = Field(ge=0, le=100)
    completed: bool
    lines: list[GeneratedStoryLine] = Field(min_length=5, max_length=5)


class GenerateImageRequest(ContractModel):
    requestId: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class GenerateImageResponse(ContractModel):
    requestId: str
    imageUrl: str
    provider: str
