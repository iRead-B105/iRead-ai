from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PronunciationWordResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resultIndex: int = Field(ge=0)
    word: str = Field(min_length=1)
    accuracyScore: float | None = Field(default=None, ge=0, le=100)
    errorType: str = Field(min_length=1)
    offsetMs: int = Field(ge=0)
    durationMs: int = Field(ge=0)


class PronunciationAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1)
    pronunciationAccuracyScore: float = Field(ge=0, le=100)
    fluencyScore: float | None = Field(default=None, ge=0, le=100)
    completenessScore: float | None = Field(default=None, ge=0, le=100)
    pronScore: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    analysisVersion: str = Field(min_length=1)
    words: list[PronunciationWordResult] = Field(min_length=1)
