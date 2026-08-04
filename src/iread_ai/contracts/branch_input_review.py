from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

BranchReviewDecision = Literal["ALLOW", "CONFIRM", "RETRY", "BLOCK"]
BranchReviewReason = Literal[
    "OK",
    "AMBIGUOUS",
    "OFF_TOPIC",
    "SELF_HARM",
    "SEXUAL",
    "SEVERE_VIOLENCE",
    "THREAT",
    "HATE_HARASSMENT",
    "PII",
    "INJECTION",
]


class BranchInputReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: StrictStr = Field(alias="requestId", min_length=1)
    question: StrictStr = Field(min_length=1)
    options: list[StrictStr] = Field(min_length=3, max_length=3)
    transcript: StrictStr = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def validate_options(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("options must contain nonblank text")
        return values


class BranchInputReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: StrictStr = Field(alias="requestId", min_length=1)
    decision: BranchReviewDecision
    reason_code: BranchReviewReason = Field(alias="reasonCode")
    policy_version: StrictStr = Field(alias="policyVersion", min_length=1)


__all__ = [
    "BranchInputReviewRequest",
    "BranchInputReviewResponse",
    "BranchReviewDecision",
    "BranchReviewReason",
]
