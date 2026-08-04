from __future__ import annotations

import pytest

from iread_ai.application.branch_input_review import DeterministicBranchInputReviewer
from iread_ai.contracts.branch_input_review import BranchInputReviewRequest


def request(transcript: str) -> BranchInputReviewRequest:
    return BranchInputReviewRequest.model_validate(
        {
            "requestId": "review-1",
            "question": "토끼는 이제 무엇을 할까요?",
            "options": ["다리를 건너요", "친구를 불러요", "숲으로 돌아가요"],
            "transcript": transcript,
        }
    )


@pytest.mark.asyncio
async def test_safe_free_form_answer_is_allowed_without_rewriting() -> None:
    reviewer = DeterministicBranchInputReviewer()

    result = await reviewer.review(request("강을 따라가 볼래요"))

    assert result.decision == "ALLOW"
    assert result.reason_code == "OK"
    assert not hasattr(result, "corrected_transcript")


@pytest.mark.asyncio
async def test_ambiguous_short_answer_requires_confirmation() -> None:
    result = await DeterministicBranchInputReviewer().review(request("음"))

    assert result.decision == "CONFIRM"
    assert result.reason_code == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_high_risk_answer_is_blocked() -> None:
    result = await DeterministicBranchInputReviewer().review(request("목을 잘라 버릴래요"))

    assert result.decision == "BLOCK"
    assert result.reason_code == "SEVERE_VIOLENCE"


@pytest.mark.asyncio
async def test_personal_identifier_is_blocked() -> None:
    result = await DeterministicBranchInputReviewer().review(
        request("우리 집 전화번호는 010-1234-5678이야")
    )

    assert result.decision == "BLOCK"
    assert result.reason_code == "PII"
