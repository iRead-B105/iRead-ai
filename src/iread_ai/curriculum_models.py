from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CurriculumContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReadingFeatureCategory = Literal[
    "GRAPHEME",
    "SYLLABLE",
    "PHONOLOGY",
    "WORD",
    "SENTENCE",
]
DataSufficiency = Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
RecommendationProvider = Literal["gms", "deterministic", "deterministic-fallback"]
RecommendationRole = Literal["CORE", "REINFORCEMENT", "STRETCH"]
CandidateStatus = Literal["ELIGIBLE", "BLOCKED"]
RecommendationReasonCode = Literal[
    "HIGH_WEAKNESS",
    "LOW_ACCURACY",
    "RELIABLE_EVIDENCE",
    "GAZE_BURDEN",
    "CURRENT_STAGE_MATCH",
    "FOUNDATION_REVIEW",
    "NEXT_STAGE_SCAFFOLD",
    "LIMITED_EVIDENCE",
    "RECENT_REPEAT_PENALTY",
]


class CurriculumFeatureProfile(CurriculumContractModel):
    featureCode: str = Field(min_length=1, max_length=150)
    category: ReadingFeatureCategory | None = None
    accuracyRate: float = Field(ge=0, le=1)
    weaknessScore: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidenceCount: int = Field(ge=0)
    pronunciationErrorRate: float | None = Field(default=None, ge=0, le=1)
    avgFixationDurationMs: int | None = Field(default=None, ge=0, le=10_000)
    avgRegressionCount: float | None = Field(default=None, ge=0, le=100)
    skipRate: float | None = Field(default=None, ge=0, le=1)


class RecentCurriculumTraining(CurriculumContractModel):
    trainingTemplateId: int = Field(ge=1, le=34)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    daysAgo: int = Field(default=0, ge=0, le=3650)


class CurriculumRecommendRequest(CurriculumContractModel):
    requestId: str = Field(min_length=1, max_length=128)
    schemaVersion: int = Field(default=1, ge=1)
    featureProfiles: list[CurriculumFeatureProfile] = Field(
        default_factory=list,
        max_length=200,
    )
    recentTrainings: list[RecentCurriculumTraining] = Field(
        default_factory=list,
        max_length=100,
    )
    currentStageHint: int | None = Field(default=None, ge=1, le=8)
    useLlm: bool = True

    @model_validator(mode="after")
    def validate_unique_profiles(self) -> CurriculumRecommendRequest:
        feature_codes = [profile.featureCode for profile in self.featureProfiles]
        if len(feature_codes) != len(set(feature_codes)):
            raise ValueError("featureProfiles must use unique featureCode values")
        return self


class CurriculumRecommendation(CurriculumContractModel):
    sequenceNo: int = Field(ge=1, le=5)
    trainingTemplateId: int = Field(ge=1, le=34)
    trainingType: str
    trainingName: str
    curriculumStage: int = Field(ge=1, le=8)
    role: RecommendationRole
    recommendedDifficulty: int = Field(ge=1, le=5)
    score: float = Field(ge=0, le=1)
    targetFeatureCodes: list[str] = Field(max_length=3)
    reasonCodes: list[RecommendationReasonCode] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=240)


class CurriculumCandidateAudit(CurriculumContractModel):
    trainingTemplateId: int = Field(ge=1, le=34)
    trainingType: str
    trainingName: str
    curriculumStage: int = Field(ge=1, le=8)
    status: CandidateStatus
    score: float | None = Field(default=None, ge=0, le=1)
    reasonCode: str


class CurriculumRecommendResponse(CurriculumContractModel):
    requestId: str
    schemaVersion: int
    recommendationVersion: Literal["CURRICULUM_HYBRID_V1"] = "CURRICULUM_HYBRID_V1"
    recommendationProvider: RecommendationProvider
    dataSufficiency: DataSufficiency
    currentStage: int = Field(ge=1, le=8)
    maximumAllowedStage: int = Field(ge=1, le=8)
    stageRationale: str
    recommendations: list[CurriculumRecommendation] = Field(min_length=5, max_length=5)
    candidateAudit: list[CurriculumCandidateAudit] = Field(min_length=34, max_length=34)
    warnings: list[str]

    @model_validator(mode="after")
    def validate_recommendation_shape(self) -> CurriculumRecommendResponse:
        if [item.sequenceNo for item in self.recommendations] != [1, 2, 3, 4, 5]:
            raise ValueError("recommendation sequenceNo values must be 1 through 5")
        template_ids = [item.trainingTemplateId for item in self.recommendations]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("recommendations must use five unique training templates")
        roles = [item.role for item in self.recommendations]
        if roles.count("CORE") != 3:
            raise ValueError("recommendations must contain three CORE trainings")
        if roles.count("REINFORCEMENT") != 1 or roles.count("STRETCH") != 1:
            raise ValueError(
                "recommendations must contain one REINFORCEMENT and one STRETCH training"
            )
        if any(item.curriculumStage > self.maximumAllowedStage for item in self.recommendations):
            raise ValueError("recommendation exceeds maximumAllowedStage")
        return self


class LlmCurriculumSelection(CurriculumContractModel):
    trainingTemplateId: int = Field(ge=1, le=34)
    role: RecommendationRole
    reasonCodes: list[RecommendationReasonCode] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=240)


class LlmCurriculumDraft(CurriculumContractModel):
    selections: list[LlmCurriculumSelection] = Field(min_length=5, max_length=5)
