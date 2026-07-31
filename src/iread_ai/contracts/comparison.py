from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    StringConstraints,
    model_validator,
)

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)

NonEmptyText = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ComparisonContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda name: "".join(
            [
                name.split("_")[0],
                *[part.title() for part in name.split("_")[1:]],
            ]
        ),
        populate_by_name=True,
        extra="forbid",
    )


class DisplayedChapterComparisonRequest(ComparisonContractModel):
    request_id: NonEmptyText
    chapter_request: StoryChapterGenerateRequest
    personalized_response: StoryChapterGenerateResponse

    @model_validator(mode="after")
    def validate_chapter_pair(self) -> DisplayedChapterComparisonRequest:
        self.personalized_response.validate_against_request(
            self.chapter_request,
        )
        return self


__all__ = ["DisplayedChapterComparisonRequest"]
