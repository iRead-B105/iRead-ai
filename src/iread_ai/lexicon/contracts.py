from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LexiconContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LexiconTargetFeature(LexiconContract):
    featureCode: str = Field(min_length=1, max_length=128)
    weaknessScore: float = Field(default=1, ge=0, le=1)
    confidence: float = Field(default=1, ge=0, le=1)


class LexiconPaletteRequest(LexiconContract):
    requestId: str = Field(min_length=1, max_length=128)
    schemaVersion: int = Field(default=1, ge=1, le=1)
    targetFeatures: list[LexiconTargetFeature] = Field(default_factory=list, max_length=8)
    excludedFeatures: list[str] = Field(default_factory=list, max_length=32)
    masteredFeatures: list[str] = Field(default_factory=list, max_length=64)
    semanticTags: list[str] = Field(default_factory=list, max_length=12)
    partsOfSpeech: list[str] = Field(default_factory=list, max_length=12)
    minSyllables: int = Field(default=1, ge=1, le=10)
    maxSyllables: int = Field(default=4, ge=1, le=10)
    maxBatchimRatio: float = Field(default=1, ge=0, le=1)
    strictPronunciation: bool = True
    requireTarget: bool = False
    includeInflections: bool = False
    limit: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def validate_policy(self) -> LexiconPaletteRequest:
        if self.minSyllables > self.maxSyllables:
            raise ValueError("minSyllables must not exceed maxSyllables")
        target_codes = [item.featureCode.strip() for item in self.targetFeatures]
        excluded = [item.strip() for item in self.excludedFeatures]
        mastered = [item.strip() for item in self.masteredFeatures]
        for name, values in (
            ("targetFeatures", target_codes),
            ("excludedFeatures", excluded),
            ("masteredFeatures", mastered),
            ("semanticTags", self.semanticTags),
            ("partsOfSpeech", self.partsOfSpeech),
        ):
            if any(not value.strip() for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty values")
        if set(target_codes).intersection(excluded):
            raise ValueError("targetFeatures and excludedFeatures must not overlap")
        return self


class LexiconItem(LexiconContract):
    formId: int
    headword: str
    surface: str
    pronunciation: str | None
    partOfSpeech: str
    definition: str
    storyTier: str
    semanticTags: list[str]
    syllableCount: int
    batchimCount: int
    batchimRatio: float
    pronunciationStatus: str
    features: dict[str, int]
    score: float
    reasons: list[str]


class LexiconPaletteResponse(LexiconContract):
    requestId: str
    schemaVersion: int = 1
    databaseVersion: str
    analyzerVersion: str
    items: list[LexiconItem]


class LexiconStatusResponse(LexiconContract):
    status: str
    databasePath: str
    databaseVersion: str | None = None
    analyzerVersion: str | None = None
    lexemeCount: int = 0
    formCount: int = 0
    featureCount: int = 0
    reviewRequiredCount: int = 0
    reason: str | None = None


__all__ = [
    "LexiconItem",
    "LexiconPaletteRequest",
    "LexiconPaletteResponse",
    "LexiconStatusResponse",
    "LexiconTargetFeature",
]
