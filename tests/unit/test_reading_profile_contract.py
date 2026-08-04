from __future__ import annotations

import pytest
from pydantic import ValidationError

from iread_ai.contracts.reading_profile import ReadingFeatureProfile
from iread_ai.curriculum_models import CurriculumRecommendRequest


def _profile_payload() -> dict[str, object]:
    return {
        "featureCode": "SYLLABLE.COMPLEX_CODA",
        "accuracyRate": 0.42,
        "avgPronunciationScore": 54.0,
        "pronunciationErrorRate": 0.46,
        "avgFixationDurationMs": 1350,
        "avgFixationCount": 3.1,
        "avgRegressionCount": 2.4,
        "skipRate": 0.18,
        "avgReadingTimeMs": 2800,
        "weaknessScore": 0.73,
        "confidence": 0.82,
        "evidenceCount": 15,
    }


def test_reading_profile_accepts_backend_normalized_shape() -> None:
    profile = ReadingFeatureProfile.model_validate(_profile_payload())

    assert profile.feature_code == "SYLLABLE.COMPLEX_CODA"
    assert profile.avg_pronunciation_score == 54.0
    assert profile.weakness_score == 0.73


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracyRate", 42.0),
        ("weaknessScore", 730.0),
        ("confidence", 82.0),
        ("avgPronunciationScore", 540.0),
    ],
)
def test_reading_profile_rejects_non_normalized_scores(field: str, value: float) -> None:
    payload = _profile_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        ReadingFeatureProfile.model_validate(payload)


@pytest.mark.parametrize(
    "feature_code",
    ["HAS_COMPLEX_CODA", "SYLLABLE", "UNKNOWN.VALUE", "SYLLABLE..CVC"],
)
def test_reading_profile_rejects_non_backend_feature_codes(feature_code: str) -> None:
    payload = _profile_payload()
    payload["featureCode"] = feature_code

    with pytest.raises(ValidationError, match="Backend reading feature namespace"):
        ReadingFeatureProfile.model_validate(payload)


def test_curriculum_uses_the_shared_reading_profile_contract() -> None:
    request = CurriculumRecommendRequest.model_validate(
        {
            "requestId": "shared-profile-contract",
            "schemaVersion": 1,
            "featureProfiles": [_profile_payload()],
            "recentTrainings": [],
            "useLlm": False,
        }
    )

    profile = request.featureProfiles[0]
    assert profile.feature_code == "SYLLABLE.COMPLEX_CODA"
    assert profile.avg_pronunciation_score == 54.0
    assert profile.weakness_score == 0.73


def test_curriculum_rejects_category_that_disagrees_with_feature_code() -> None:
    profile = _profile_payload()
    profile["category"] = "WORD"

    with pytest.raises(ValidationError, match="category must match"):
        CurriculumRecommendRequest.model_validate(
            {
                "requestId": "mismatched-feature-category",
                "schemaVersion": 1,
                "featureProfiles": [profile],
                "recentTrainings": [],
                "useLlm": False,
            }
        )
