"""훈련·이야기·이미지 생성 API 계약 모델."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class GeneratedStoryLine(ContractModel):
    content: str
    requiresBranchInput: bool


class GenerateStoryResponse(ContractModel):
    requestId: str
    schemaVersion: int
    nextProgress: int = Field(ge=0, le=100)
    completed: bool
    lines: list[GeneratedStoryLine]


class GenerateImageRequest(ContractModel):
    requestId: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class GenerateImageResponse(ContractModel):
    requestId: str
    imageUrl: str
    provider: str
