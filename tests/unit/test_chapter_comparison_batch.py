from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.devtools.chapter_comparison_batch import (
    PROFILE_KEYS,
    STAGE_LABELS,
    _validate_real_provider_health,
    aggregate_records,
    build_cases,
    normalized_chapter_indexes,
    numeric_statistics,
)
from iread_ai.devtools.service_story_catalog import STORY_CATALOG


def _context_without_profile(
    request: dict[str, Any],
) -> dict[str, Any]:
    value = deepcopy(request)
    value.pop("requestId")
    value.pop("generationProfile")
    return value


def test_build_cases_produces_balanced_100_case_matrix() -> None:
    cases = build_cases(run_token="unit")

    assert len(cases) == 100
    assert len({case["caseId"] for case in cases}) == 100
    assert {case["profileKey"] for case in cases} == set(PROFILE_KEYS)
    assert {case["stage"] for case in cases} == set(STAGE_LABELS)
    assert {
        profile: sum(case["profileKey"] == profile for case in cases) for profile in PROFILE_KEYS
    } == {"balanced": 50, "beginner": 50}
    assert {stage: sum(case["stage"] == stage for case in cases) for stage in STAGE_LABELS} == {
        stage: 20 for stage in STAGE_LABELS
    }
    assert {
        story.template_id: sum(case["storyId"] == story.template_id for case in cases)
        for story in STORY_CATALOG
    } == {story.template_id: 10 for story in STORY_CATALOG}


@pytest.mark.parametrize("provider", ["gms", "openai"])
def test_batch_preflight_accepts_real_story_providers(
    provider: str,
) -> None:
    assert _validate_real_provider_health({"status": "UP", "storyProvider": provider}) == provider


def test_batch_preflight_rejects_mock_provider() -> None:
    with pytest.raises(
        RuntimeError,
        match="storyProvider=gms or openai",
    ):
        _validate_real_provider_health({"status": "ok", "storyProvider": "mock"})


def test_each_story_uses_five_unique_normalized_positions() -> None:
    expected = {
        5: (0, 1, 2, 3, 4),
        6: (0, 1, 2, 4, 5),
        7: (0, 2, 3, 4, 6),
        8: (0, 2, 4, 5, 7),
    }

    for story in STORY_CATALOG:
        indexes = normalized_chapter_indexes(story)
        assert indexes == expected[story.total_chapters]
        assert len(set(indexes)) == 5
        assert indexes[0] == 0
        assert indexes[-1] == story.total_chapters - 1


def test_profiles_share_exact_context_and_child_input() -> None:
    cases = build_cases(run_token="same-context")
    pairs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for case in cases:
        pairs.setdefault(
            (case["storyId"], case["stage"]),
            [],
        ).append(case)

    assert len(pairs) == 50
    for pair in pairs.values():
        assert len(pair) == 2
        left, right = pair
        assert left["childInput"] == right["childInput"]
        assert left["inputType"] == right["inputType"]
        assert _context_without_profile(left["chapterRequest"]) == _context_without_profile(
            right["chapterRequest"]
        )


def test_every_synthetic_request_validates_current_v3_contract() -> None:
    cases = build_cases(run_token="contract")

    for case in cases:
        request = StoryChapterGenerateRequest.model_validate(case["chapterRequest"])
        assert request.schema_version == 3
        assert request.chapter_number == case["chapterNumber"]
        assert request.story_revision == case["chapterNumber"] - 1
        if case["stage"] == "opening":
            assert request.branch_input is None
            assert request.story_state.last_question is None
        else:
            assert request.branch_input is not None
            assert request.story_state.last_question
            assert request.story_state.recent_pages
        if case["stage"] == "ending":
            assert request.conclude is True
            assert request.chapter_plan.question_focus is None


def _outcome(
    *,
    score: float,
    risk: float,
    contract_pass: bool,
    analysis_status: str,
    text: str,
    model_calls: int,
) -> dict[str, Any]:
    return {
        "pages": [
            {
                "pageNumber": 1,
                "sentences": [text],
                "question": None,
                "choices": [],
                "requiresBranchInput": False,
            }
        ],
        "fit": {
            "profileFitScore": score,
            "surfaceProfileFitScore": score,
            "riskPer10": risk,
            "surfaceRiskPer10": risk,
            "contractPass": contract_pass,
            "contractFailures": ([] if contract_pass else ["WRITTEN_SYLLABLE_RANGE"]),
            "analysisStatus": analysis_status,
            "unverifiedSkillCodes": (["PHONO_LIAISON"] if analysis_status != "FULL" else []),
        },
        "generation": {
            "apiCallCount": model_calls,
            "candidateCount": 3 if model_calls > 1 else 1,
            "repairAttempted": model_calls > 1,
            "repairAccepted": model_calls > 1,
        },
        "timingMs": {"total": 1000.0 + score},
    }


def _record(
    *,
    case_id: str,
    profile: str,
    stage: str,
    basis: str,
    plain_score: float,
    personalized_score: float,
) -> dict[str, Any]:
    score_delta = personalized_score - plain_score
    confidence = "FULL" if basis == "FULL_POLICY" else "PARTIAL"
    plain = _outcome(
        score=plain_score,
        risk=2.0,
        contract_pass=True,
        analysis_status=confidence,
        text=f"{case_id} 일반 글",
        model_calls=1,
    )
    personalized = _outcome(
        score=personalized_score,
        risk=1.0,
        contract_pass=True,
        analysis_status=confidence,
        text=f"{case_id} 개인화 글",
        model_calls=2,
    )
    return {
        "case": {
            "caseId": case_id,
            "storyId": 1001,
            "storyTitle": "토끼와 거북이",
            "profileKey": profile,
            "stage": stage,
            "chapterNumber": 2,
            "inputType": "CHOICE_ON_TOPIC",
            "childInput": "힘내!",
        },
        "status": "SUCCESS",
        "personalized": {"response": {}, "elapsedMs": 1000.0},
        "comparison": {
            "response": {
                "plain": plain,
                "personalized": personalized,
                "comparison": {
                    "scoreBasis": basis,
                    "comparisonConfidence": confidence,
                    "winner": (
                        "PERSONALIZED"
                        if score_delta > 0.005
                        else "PLAIN"
                        if score_delta < -0.005
                        else "TIE"
                    ),
                    "plainProfileFitScore": plain_score,
                    "personalizedProfileFitScore": personalized_score,
                    "delta": {
                        "profileFitScore": score_delta,
                        "riskPer10": -1.0,
                        "excludedOverage": -1,
                        "limitedOverage": 0,
                        "targetDistance": 0,
                        "totalElapsedMs": 200.0,
                    },
                },
                "diagnostics": {"newApiCallCount": 1},
            },
            "elapsedMs": 800.0,
        },
        "error": None,
    }


def test_numeric_statistics_include_high_low_percentiles_and_bootstrap() -> None:
    stats = numeric_statistics([0.0, 10.0, 20.0, 30.0], seed=17)

    assert stats["n"] == 4
    assert stats["min"] == 0.0
    assert stats["max"] == 30.0
    assert stats["mean"] == 15.0
    assert stats["median"] == 15.0
    assert stats["p05"] == 1.5
    assert stats["p95"] == 28.5
    assert len(stats["bootstrapMean95"]) == 2


def test_aggregate_separates_full_and_partial_and_keeps_case_text() -> None:
    records = [
        _record(
            case_id="high",
            profile="beginner",
            stage="early",
            basis="FULL_POLICY",
            plain_score=20.0,
            personalized_score=50.0,
        ),
        _record(
            case_id="middle",
            profile="balanced",
            stage="middle",
            basis="COMMON_SURFACE_ONLY",
            plain_score=40.0,
            personalized_score=50.0,
        ),
        _record(
            case_id="tie",
            profile="balanced",
            stage="late",
            basis="FULL_POLICY",
            plain_score=60.0,
            personalized_score=60.0,
        ),
        _record(
            case_id="low",
            profile="beginner",
            stage="ending",
            basis="COMMON_SURFACE_ONLY",
            plain_score=70.0,
            personalized_score=50.0,
        ),
    ]

    summary, rows = aggregate_records(
        records,
        seed=23,
        elapsed_seconds=12.5,
    )

    assert len(rows) == 4
    assert summary["imageApiCallCount"] == 0
    assert summary["modelCallCount"] == 12
    assert summary["overall"]["scoreBasisCounts"] == {
        "FULL_POLICY": 2,
        "COMMON_SURFACE_ONLY": 2,
    }
    assert summary["overall"]["winTieLoss"] == {
        "personalizedWins": 2,
        "ties": 1,
        "plainWins": 1,
        "personalizedWinRate": 0.5,
        "exactSignP": 1.0,
    }
    assert summary["overall"]["repair"] == {
        "attempted": 4,
        "accepted": 4,
        "attemptRate": 1.0,
        "acceptRateAmongAttempts": 1.0,
    }
    score = summary["overall"]["metrics"]["profileFitScoreDelta"]
    assert score["min"] == -20.0
    assert score["max"] == 30.0
    assert score["mean"] == 5.0
    assert summary["overall"]["highestCase"]["caseId"] == "high"
    assert "개인화 글" in summary["overall"]["highestCase"]["personalizedText"]
    assert summary["overall"]["lowestCase"]["caseId"] == "low"
    assert set(summary["byScoreBasis"]) == {
        "FULL_POLICY",
        "COMMON_SURFACE_ONLY",
    }
