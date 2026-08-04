from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)

UnitRate = Annotated[StrictFloat, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
PronunciationScore = Annotated[StrictFloat, Field(ge=0, le=100)]
ReadingFeatureCode = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=150),
]
ProfileAnalysisVersion = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
ProfileStatus = Literal["NORMAL", "WATCH", "WEAK", "CRITICAL"]

READING_FEATURE_CATEGORIES = frozenset(
    {"GRAPHEME", "SYLLABLE", "PHONOLOGY", "WORD", "SENTENCE"}
)


def _to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return "".join((head, *(part.title() for part in tail)))


class ReadingProfileContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=False,
        extra="forbid",
    )


class ReadingFeatureProfile(ReadingProfileContractModel):
    """Backend-compatible aggregate reading feature profile.

    Rates use the normalized 0..1 range and pronunciation scores use 0..100.
    Feature codes follow the Backend reading_features namespace.
    """

    feature_code: ReadingFeatureCode
    accuracy_rate: UnitRate
    avg_pronunciation_score: PronunciationScore | None = None
    pronunciation_error_rate: UnitRate | None = None
    avg_fixation_duration_ms: StrictInt | None = Field(default=None, ge=0)
    avg_fixation_count: NonNegativeFloat | None = None
    avg_regression_count: NonNegativeFloat | None = None
    skip_rate: UnitRate | None = None
    avg_reading_time_ms: StrictInt | None = Field(default=None, ge=0)
    weakness_score: UnitRate
    confidence: UnitRate
    evidence_count: StrictInt = Field(ge=0)

    @field_validator("feature_code")
    @classmethod
    def validate_backend_feature_code(cls, value: str) -> str:
        category, separator, detail = value.partition(".")
        if (
            not separator
            or category not in READING_FEATURE_CATEGORIES
            or not detail
            or any(not segment or segment.isspace() for segment in detail.split("."))
        ):
            raise ValueError(
                "featureCode must use a Backend reading feature namespace: "
                "GRAPHEME.*, SYLLABLE.*, PHONOLOGY.*, WORD.*, or SENTENCE.*"
            )
        return value


class BackendStudentFeatureProfileView(ReadingFeatureProfile):
    """Exact AI-side contract for Backend StudentFeatureProfileView."""

    skip_rate: UnitRate
    status: ProfileStatus
    analysis_version: ProfileAnalysisVersion
    analyzed_at: datetime


class StudentReadingProfileSnapshot(ReadingProfileContractModel):
    feature_profiles: list[BackendStudentFeatureProfileView] = Field(max_length=200)

    @model_validator(mode="after")
    def validate_snapshot_consistency(self) -> Self:
        feature_codes = [profile.feature_code for profile in self.feature_profiles]
        if len(feature_codes) != len(set(feature_codes)):
            raise ValueError("featureProfiles featureCode values must be unique")
        versions = {profile.analysis_version for profile in self.feature_profiles}
        if len(versions) > 1:
            raise ValueError("featureProfiles analysisVersion values must match")
        return self

    @property
    def profile_analysis_version(self) -> str:
        if not self.feature_profiles:
            return "WEAKNESS_V1"
        return self.feature_profiles[0].analysis_version


__all__ = [
    "NonNegativeFloat",
    "BackendStudentFeatureProfileView",
    "ProfileAnalysisVersion",
    "ProfileStatus",
    "PronunciationScore",
    "READING_FEATURE_CATEGORIES",
    "ReadingFeatureCode",
    "ReadingFeatureProfile",
    "ReadingProfileContractModel",
    "StudentReadingProfileSnapshot",
    "UnitRate",
]
