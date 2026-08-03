from __future__ import annotations

from datetime import datetime
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

from iread_ai.contracts.story_page import NonEmptyText, ShortIdentifier

UnitRate = Annotated[StrictFloat, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
SummaryText = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
GazeSeriesStatus = Literal["AVAILABLE", "NO_DATA", "FAILED"]
DataSufficiency = Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
SummaryProvider = Literal["deterministic", "gms", "deterministic-fallback"]


def _to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return "".join((head, *(part.title() for part in tail)))


class TeacherReportContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=False,
        extra="forbid",
    )


class TeacherReportFeatureProfile(TeacherReportContractModel):
    feature_code: ShortIdentifier
    feature_label: NonEmptyText = Field(max_length=120)
    accuracy_rate: UnitRate
    avg_pronunciation_score: StrictInt | None = Field(default=None, ge=0, le=1000)
    pronunciation_error_rate: UnitRate | None = None
    avg_fixation_duration_ms: StrictInt | None = Field(default=None, ge=0)
    avg_fixation_count: NonNegativeFloat | None = None
    avg_regression_count: NonNegativeFloat | None = None
    skip_rate: UnitRate
    avg_reading_time_ms: StrictInt | None = Field(default=None, ge=0)
    weakness_score: StrictInt = Field(ge=0, le=1000)
    confidence: UnitRate
    evidence_count: StrictInt = Field(ge=0)
    previous_accuracy_rate: UnitRate | None = None
    previous_weakness_score: StrictInt | None = Field(default=None, ge=0, le=1000)


class TeacherReportGazePoint(TeacherReportContractModel):
    observed_at: datetime
    total_visited_duration_ms: StrictInt = Field(ge=0)
    total_visited_count: StrictInt = Field(ge=0)
    reverse_read_count: StrictInt = Field(ge=0)
    avg_visited_duration_ms: StrictInt | None = Field(default=None, ge=0)


class TeacherReportGazeSeries(TeacherReportContractModel):
    status: GazeSeriesStatus
    comparison_available: StrictBool
    points: list[TeacherReportGazePoint] = Field(max_length=100)
    failed_session_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_status_and_points(self) -> Self:
        if self.status == "AVAILABLE" and not self.points:
            raise ValueError("AVAILABLE gaze series requires at least one point")
        if self.status != "AVAILABLE" and self.points:
            raise ValueError("non-AVAILABLE gaze series must not contain points")
        if self.status == "FAILED" and self.failed_session_count == 0:
            raise ValueError("FAILED gaze series requires a failed session")
        if self.status == "NO_DATA" and self.failed_session_count != 0:
            raise ValueError("NO_DATA gaze series must not contain failed sessions")
        if self.comparison_available != (len(self.points) >= 2):
            raise ValueError(
                "comparisonAvailable must be true exactly when two or more points exist"
            )
        return self


class TeacherReportGazeTrend(TeacherReportContractModel):
    training: TeacherReportGazeSeries
    test: TeacherReportGazeSeries


class TeacherReportAnalyzeRequest(TeacherReportContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[1]
    profile_analysis_version: ShortIdentifier
    feature_profiles: list[TeacherReportFeatureProfile] = Field(max_length=100)
    gaze_trend: TeacherReportGazeTrend

    @model_validator(mode="after")
    def validate_unique_features(self) -> Self:
        codes = [profile.feature_code for profile in self.feature_profiles]
        if len(codes) != len(set(codes)):
            raise ValueError("featureProfiles featureCode values must be unique")
        return self


class TeacherReportGazeDescriptions(TeacherReportContractModel):
    training: list[SummaryText] = Field(max_length=5)
    test: list[SummaryText] = Field(max_length=5)


class TeacherReportAnalyzeResponse(TeacherReportContractModel):
    request_id: ShortIdentifier
    schema_version: Literal[1]
    analysis_version: ShortIdentifier
    summary_provider: SummaryProvider
    data_sufficiency: DataSufficiency
    improved_patterns: list[SummaryText] = Field(max_length=5)
    persistent_difficulty_patterns: list[SummaryText] = Field(max_length=5)
    gaze_descriptions: TeacherReportGazeDescriptions

    def validate_against_request(
        self,
        request: TeacherReportAnalyzeRequest,
    ) -> Self:
        if self.request_id != request.request_id:
            raise ValueError("response requestId must match request requestId")
        return self


class TeacherReportNarrativeItem(TeacherReportContractModel):
    text: SummaryText = Field(
        description=(
            "교수자에게 표시할 관찰 문장. evidenceId, ID 값, 대괄호 인용을 포함하지 않는다."
        )
    )
    evidence_ids: list[ShortIdentifier] = Field(
        min_length=1,
        max_length=3,
        description="문장의 근거 ID. 근거 ID는 이 배열에만 기록한다.",
    )

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> Self:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidenceIds must be unique")
        return self


class TeacherReportNarrativeDraft(TeacherReportContractModel):
    improved_patterns: list[TeacherReportNarrativeItem] = Field(max_length=5)
    persistent_difficulty_patterns: list[TeacherReportNarrativeItem] = Field(max_length=5)
    training_gaze_descriptions: list[TeacherReportNarrativeItem] = Field(max_length=5)
    test_gaze_descriptions: list[TeacherReportNarrativeItem] = Field(max_length=5)


__all__ = [
    "DataSufficiency",
    "SummaryProvider",
    "TeacherReportAnalyzeRequest",
    "TeacherReportAnalyzeResponse",
    "TeacherReportFeatureProfile",
    "TeacherReportGazeDescriptions",
    "TeacherReportGazePoint",
    "TeacherReportGazeSeries",
    "TeacherReportGazeTrend",
    "TeacherReportNarrativeDraft",
    "TeacherReportNarrativeItem",
]
