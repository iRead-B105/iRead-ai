from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=False,
    )

    code: StrictStr
    message: StrictStr
    request_id: StrictStr = Field(alias="requestId")
    retryable: StrictBool | None = None


__all__ = ["ErrorResponse"]
