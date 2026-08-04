from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
)

UnitRate = Annotated[StrictFloat, Field(ge=0, le=1)]
NonNegativeFloat = Annotated[StrictFloat, Field(ge=0)]
PronunciationScore = Annotated[StrictFloat, Field(ge=0, le=100)]
ReadingFeatureCode = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=150),
]

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


__all__ = [
    "NonNegativeFloat",
    "PronunciationScore",
    "READING_FEATURE_CATEGORIES",
    "ReadingFeatureCode",
    "ReadingFeatureProfile",
    "ReadingProfileContractModel",
    "UnitRate",
]
