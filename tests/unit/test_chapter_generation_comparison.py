from __future__ import annotations

from copy import deepcopy

import pytest

from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.personalization.chapter_comparison import (
    ChapterGenerationComparisonService,
)
from tests.unit.test_story_chapter_contracts import (
    request_payload,
    response_payload,
)


class RecordingBaselineService:
    def __init__(self, response: StoryChapterGenerateResponse) -> None:
        self.response = response
        self.calls: list[StoryChapterGenerateRequest] = []

    async def generate(
        self,
        request: StoryChapterGenerateRequest,
    ) -> StoryChapterGenerateResponse:
        self.calls.append(request)
        return self.response


def _baseline_response() -> StoryChapterGenerateResponse:
    payload = response_payload()
    payload["generation"].update(
        {
            "promptVersion": "chapter-baseline-test",
            "candidateCount": 1,
            "selectedCandidateId": "baseline-1",
            "apiCallCount": 1,
            "repairAttempted": False,
            "repairAccepted": False,
        }
    )
    payload["timingMs"].update(
        {
            "generation": 1800.0,
            "analysis": 180.0,
            "pagination": 1.0,
            "repair": 0.0,
            "total": 1981.0,
        }
    )
    payload["quality"]["chapter"].update(
        {
            "riskPer10": 1.0,
            "perSkill": [
                {
                    "code": "ONSET_ㄲ",
                    "role": "LIMITED",
                    "status": "OVER_LIMIT",
                    "occurrences": 3,
                    "maxOccurrences": 1,
                    "targetMin": None,
                    "targetMax": None,
                    "overage": 2,
                    "targetDistance": None,
                    "weightedRisk": 3.6,
                }
            ],
        }
    )
    return StoryChapterGenerateResponse.model_validate(payload)


@pytest.mark.asyncio
async def test_comparison_reuses_displayed_result_and_calls_baseline_once() -> None:
    request = StoryChapterGenerateRequest.model_validate(request_payload())
    personalized = StoryChapterGenerateResponse.model_validate(response_payload())
    baseline_service = RecordingBaselineService(_baseline_response())
    service = ChapterGenerationComparisonService(baseline_service=baseline_service)

    result = await service.compare_displayed_chapter_to_plain(
        request,
        personalized,
    )

    assert baseline_service.calls == [request]
    assert result["plain"]["generation"]["candidateCount"] == 1
    assert result["plain"]["generation"]["repairAttempted"] is False
    assert result["personalized"]["generation"]["candidateCount"] == 3
    assert result["plain"]["fit"]["featureOccurrences"]["ONSET_ㄲ"] == 3
    assert result["comparison"]["comparable"] is True
    assert result["comparison"]["winner"] == "PERSONALIZED"
    assert result["comparison"]["delta"]["profileFitScore"] > 0
    assert "생성 예산 차이" in result["comparison"]["comparisonReason"]
    assert result["diagnostics"]["newApiCallCount"] == 1
    assert result["diagnostics"]["baselineApiCallCount"] == 1


@pytest.mark.asyncio
async def test_non_full_g2p_returns_a_labeled_partial_score() -> None:
    request_document = request_payload()
    request_document["generationProfile"]["skills"].append(
        {
            "code": "PHONO_LIAISON",
            "role": "EXCLUDED",
            "maxOccurrences": 0,
            "targetMin": None,
            "targetMax": None,
            "unitPenalty": 1.5,
        }
    )
    request = StoryChapterGenerateRequest.model_validate(request_document)
    baseline_document = _baseline_response().model_dump(
        mode="json",
        by_alias=True,
    )
    baseline_document["quality"]["chapter"]["analysisStatus"] = "UNRELIABLE"
    baseline_document["quality"]["chapter"]["status"] = "ANALYSIS_DEGRADED"
    baseline = StoryChapterGenerateResponse.model_validate(baseline_document)
    personalized = StoryChapterGenerateResponse.model_validate(deepcopy(response_payload()))
    service = ChapterGenerationComparisonService(
        baseline_service=RecordingBaselineService(baseline)
    )

    result = await service.compare_displayed_chapter_to_plain(
        request,
        personalized,
    )

    assert isinstance(
        result["plain"]["fit"]["profileFitScore"],
        float,
    )
    assert result["plain"]["fit"]["comparable"] is True
    assert result["plain"]["fit"]["scoreConfidence"] == "PARTIAL"
    assert result["plain"]["fit"]["unverifiedSkillCodes"] == ["PHONO_LIAISON"]
    assert result["comparison"]["comparable"] is True
    assert result["comparison"]["comparisonConfidence"] == "PARTIAL"
    assert result["comparison"]["scoreBasis"] == "COMMON_SURFACE_ONLY"
    assert (
        result["comparison"]["plainProfileFitScore"]
        == result["plain"]["fit"]["surfaceProfileFitScore"]
    )
    assert (
        result["comparison"]["personalizedProfileFitScore"]
        == result["personalized"]["fit"]["surfaceProfileFitScore"]
    )
    assert (
        result["comparison"]["personalizedProfileFitScore"]
        != result["personalized"]["fit"]["profileFitScore"]
    )
    assert result["comparison"]["winner"] == "PERSONALIZED"
    assert isinstance(
        result["comparison"]["delta"]["profileFitScore"],
        float,
    )
    assert "0회로 판정하지 않습니다" in result["comparison"]["comparisonReason"]
