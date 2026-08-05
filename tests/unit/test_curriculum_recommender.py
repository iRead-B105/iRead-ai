from __future__ import annotations

import json
from typing import Any

import pytest

from iread_ai.application.reading_profile_request_adapter import (
    build_curriculum_recommend_request,
)
from iread_ai.contracts.reading_profile import StudentReadingProfileSnapshot
from iread_ai.curriculum_models import CurriculumRecommendRequest, RecentCurriculumTraining
from iread_ai.curriculum_recommender import recommend_curriculum
from iread_ai.devtools.backend_profile_samples import long_backend_profile_sample
from iread_ai.devtools.curriculum_samples import curriculum_sample


class _FakeProvider:
    def __init__(
        self,
        document: dict[str, Any],
        *,
        provider_name: str = "gms",
    ) -> None:
        self.document = document
        self.provider_name = provider_name
        self.model = "test-model"
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.document


def _request(sample_name: str, *, use_llm: bool = False) -> CurriculumRecommendRequest:
    sample = curriculum_sample(sample_name)
    return CurriculumRecommendRequest(
        requestId=f"test-{sample_name}",
        schemaVersion=1,
        featureProfiles=sample["featureProfiles"],
        recentTrainings=sample["recentTrainings"],
        useLlm=use_llm,
    )


def test_letter_level_student_cannot_receive_passage_or_fluency_training() -> None:
    response = recommend_curriculum(_request("자모 읽기가 어려운 학생"), provider=None)

    assert response.currentStage == 1
    assert response.maximumAllowedStage == 2
    assert len(response.recommendations) == 5
    assert all(item.curriculumStage <= 2 for item in response.recommendations)
    assert 26 not in {item.trainingTemplateId for item in response.recommendations}
    short_passage = next(item for item in response.candidateAudit if item.trainingTemplateId == 26)
    assert short_passage.status == "BLOCKED"
    assert short_passage.reasonCode == "PREREQUISITE_STAGE_NOT_REACHED"


def test_recommendation_has_three_core_one_reinforcement_and_one_stretch() -> None:
    response = recommend_curriculum(_request("음절 결합이 어려운 학생"), provider=None)

    roles = [item.role for item in response.recommendations]
    assert roles.count("CORE") == 3
    assert roles.count("REINFORCEMENT") == 1
    assert roles.count("STRETCH") == 1
    assert len({item.trainingTemplateId for item in response.recommendations}) == 5
    assert {6, 14, 24}.isdisjoint(item.trainingTemplateId for item in response.recommendations)


def test_no_evidence_starts_with_conservative_foundation_plan() -> None:
    response = recommend_curriculum(_request("신규 학생 · 근거 없음"), provider=None)

    assert response.dataSufficiency == "INSUFFICIENT"
    assert response.currentStage == 1
    assert response.maximumAllowedStage == 2
    assert max(item.curriculumStage for item in response.recommendations) == 2
    assert all(item.recommendedDifficulty <= 2 for item in response.recommendations)


def test_sentence_student_can_receive_stage_eight_scaffold() -> None:
    response = recommend_curriculum(_request("문장·유창성 단계 학생"), provider=None)

    assert response.currentStage == 7
    assert response.maximumAllowedStage == 8
    assert 27 not in {item.trainingTemplateId for item in response.recommendations}
    core = [item for item in response.recommendations if item.role == "CORE"]
    assert all(item.curriculumStage in {6, 7} for item in core)
    stretch = next(item for item in response.recommendations if item.role == "STRETCH")
    assert stretch.curriculumStage == 8
    repeated = next(item for item in response.candidateAudit if item.trainingTemplateId == 27)
    assert repeated.reasonCode == "RECENT_COOLDOWN_NOT_PREFERRED"


def test_recent_templates_are_reused_when_fresh_candidates_cannot_fill_plan() -> None:
    request = _request("자모 읽기가 어려운 학생")
    request = request.model_copy(
        update={
            "recentTrainings": [
                RecentCurriculumTraining(
                    trainingTemplateId=template_id,
                    accuracy=0.5,
                    daysAgo=1,
                )
                for template_id in (1, 2, 3, 4, 5)
            ]
        }
    )

    response = recommend_curriculum(request, provider=None)

    assert len(response.recommendations) == 5
    assert all(item.curriculumStage <= 2 for item in response.recommendations)


@pytest.mark.parametrize("provider_name", ["gms", "openai"])
def test_valid_llm_reranking_is_used(provider_name: str) -> None:
    provider = _FakeProvider(
        {
            "selections": [
                {
                    "trainingTemplateId": 2,
                    "role": "CORE",
                    "reasonCodes": ["HIGH_WEAKNESS", "CURRENT_STAGE_MATCH"],
                    "rationale": "기본 자음 정확도를 보완하기 위해 먼저 배치했습니다.",
                },
                {
                    "trainingTemplateId": 1,
                    "role": "CORE",
                    "reasonCodes": ["LOW_ACCURACY", "CURRENT_STAGE_MATCH"],
                    "rationale": "기본 모음 읽기를 현재 단계에서 반복하도록 배치했습니다.",
                },
                {
                    "trainingTemplateId": 3,
                    "role": "CORE",
                    "reasonCodes": ["CURRENT_STAGE_MATCH"],
                    "rationale": "글자에서 음절로 연결되는 기초 연습으로 배치했습니다.",
                },
                {
                    "trainingTemplateId": 10,
                    "role": "REINFORCEMENT",
                    "reasonCodes": ["FOUNDATION_REVIEW"],
                    "rationale": "받침 소리를 다시 확인하는 보강 활동으로 배치했습니다.",
                },
                {
                    "trainingTemplateId": 11,
                    "role": "STRETCH",
                    "reasonCodes": ["NEXT_STAGE_SCAFFOLD"],
                    "rationale": "허용된 다음 단계에서 끝소리 읽기를 확장하도록 배치했습니다.",
                },
            ]
        },
        provider_name=provider_name,
    )

    response = recommend_curriculum(
        _request("자모 읽기가 어려운 학생", use_llm=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert response.recommendationProvider == provider_name
    assert [item.trainingTemplateId for item in response.recommendations] == [2, 1, 3, 10, 11]


def test_invalid_llm_stage_jump_uses_deterministic_fallback() -> None:
    provider = _FakeProvider(
        {
            "selections": [
                {
                    "trainingTemplateId": template_id,
                    "role": role,
                    "reasonCodes": ["CURRENT_STAGE_MATCH"],
                    "rationale": "입력 근거에 따른 추천입니다.",
                }
                for template_id, role in [
                    (1, "CORE"),
                    (2, "CORE"),
                    (26, "CORE"),
                    (4, "REINFORCEMENT"),
                    (5, "STRETCH"),
                ]
            ]
        }
    )

    response = recommend_curriculum(
        _request("자모 읽기가 어려운 학생", use_llm=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert response.recommendationProvider == "deterministic-fallback"
    assert 26 not in {item.trainingTemplateId for item in response.recommendations}
    assert response.warnings


def _long_profile_request(*, use_llm: bool) -> CurriculumRecommendRequest:
    sample = long_backend_profile_sample()
    snapshot = StudentReadingProfileSnapshot.model_validate(
        {"featureProfiles": sample["featureProfiles"]}
    )
    return build_curriculum_recommend_request(
        request_id="long-profile-stage-gate",
        snapshot=snapshot,
        recent_trainings=sample["recentTrainings"],
        use_llm=use_llm,
    )


def test_long_profile_advances_one_stage_and_keeps_foundation_as_reinforcement() -> None:
    response = recommend_curriculum(_long_profile_request(use_llm=False), provider=None)

    assert response.currentStage == 4
    assert response.maximumAllowedStage == 5
    assert "3단계까지 안정적으로 수행" in response.stageRationale
    assert "복습 활동에 함께" in response.stageRationale
    target_codes = {
        code
        for recommendation in response.recommendations
        for code in recommendation.targetFeatureCodes
    }
    assert "SYLLABLE.COMPLEX_CODA" in target_codes
    assert all(
        not code.startswith(("PHONOLOGY.", "WORD.", "SENTENCE."))
        for code in target_codes
    )
    complex_coda_training = next(
        item for item in response.recommendations if item.trainingTemplateId == 18
    )
    assert complex_coda_training.targetFeatureCodes == ["SYLLABLE.COMPLEX_CODA"]
    final_sound_training = next(
        item for item in response.recommendations if item.trainingTemplateId == 11
    )
    assert all(
        code.startswith("GRAPHEME.CODA.")
        for code in final_sound_training.targetFeatureCodes
    )
    assert all(
        "GRAPHEME." not in recommendation.rationale
        and "SYLLABLE." not in recommendation.rationale
        for recommendation in response.recommendations
    )


def test_long_profile_sends_only_reachable_stage_evidence_to_llm() -> None:
    provider = _FakeProvider({"selections": []})

    response = recommend_curriculum(
        _long_profile_request(use_llm=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert response.recommendationProvider == "deterministic-fallback"
    document = json.loads(provider.calls[0]["user_prompt"])
    evidence = document["studentEvidence"]
    assert len(evidence) <= 12
    assert all(item["profileStage"] <= 5 for item in evidence)
    assert "PHONOLOGY.LIAISON" not in {item["featureCode"] for item in evidence}
    assert "SENTENCE.FLUENCY" not in {item["featureCode"] for item in evidence}


def test_llm_cannot_replace_core_with_candidate_below_rule_baseline() -> None:
    provider = _FakeProvider(
        {
            "selections": [
                {
                    "trainingTemplateId": template_id,
                    "role": role,
                    "reasonCodes": [reason],
                    "rationale": "현재 단계와 입력 근거를 반영한 추천입니다.",
                }
                for template_id, role, reason in [
                    (18, "CORE", "CURRENT_STAGE_MATCH"),
                    (15, "CORE", "CURRENT_STAGE_MATCH"),
                    (16, "CORE", "CURRENT_STAGE_MATCH"),
                    (11, "REINFORCEMENT", "FOUNDATION_REVIEW"),
                    (20, "STRETCH", "NEXT_STAGE_SCAFFOLD"),
                ]
            ]
        }
    )

    response = recommend_curriculum(
        _long_profile_request(use_llm=True),
        provider=provider,  # type: ignore[arg-type]
    )

    assert response.recommendationProvider == "deterministic-fallback"
    assert 16 not in {item.trainingTemplateId for item in response.recommendations}


def test_tense_onset_does_not_masquerade_as_basic_stage_one_evidence() -> None:
    request = CurriculumRecommendRequest(
        requestId="explicit-feature-stage",
        schemaVersion=1,
        useLlm=False,
        featureProfiles=[
            {
                "featureCode": "GRAPHEME.VOWEL.BASIC.ㅓ",
                "category": "GRAPHEME",
                "accuracyRate": 0.66,
                "weaknessScore": 0.53,
                "confidence": 0.94,
                "evidenceCount": 21,
            },
            {
                "featureCode": "GRAPHEME.ONSET.TENSE.ㄲ",
                "category": "GRAPHEME",
                "accuracyRate": 0.525,
                "weaknessScore": 0.61,
                "confidence": 0.805,
                "evidenceCount": 12,
            },
        ],
    )

    response = recommend_curriculum(request, provider=None)

    assert response.currentStage == 1
    assert response.maximumAllowedStage == 2
    assert "1단계 읽기 요소를 아직 어려워" in response.stageRationale
    assert "GRAPHEME.ONSET.TENSE.ㄲ" not in response.stageRationale
    assert "가벼운 도전 활동" in response.stageRationale


def test_stable_advanced_evidence_is_not_dragged_down_by_one_foundation_weakness() -> None:
    request = CurriculumRecommendRequest(
        requestId="advanced-with-foundation-debt",
        schemaVersion=1,
        useLlm=False,
        featureProfiles=[
            {
                "featureCode": "SENTENCE.FLUENCY",
                "category": "SENTENCE",
                "accuracyRate": 0.84,
                "weaknessScore": 0.35,
                "confidence": 0.93,
                "evidenceCount": 22,
            },
            {
                "featureCode": "GRAPHEME.CODA.COMPLEX.ㄺ",
                "category": "GRAPHEME",
                "accuracyRate": 0.42,
                "weaknessScore": 0.73,
                "confidence": 0.91,
                "evidenceCount": 20,
            },
        ],
    )

    response = recommend_curriculum(request, provider=None)

    assert response.currentStage == 8
    assert response.maximumAllowedStage == 8
    assert "읽기 유창성을 높이는 훈련" in response.stageRationale
    assert "복습 활동에 함께" in response.stageRationale
    reinforcement = next(
        item for item in response.recommendations if item.role == "REINFORCEMENT"
    )
    assert reinforcement.curriculumStage < response.currentStage
