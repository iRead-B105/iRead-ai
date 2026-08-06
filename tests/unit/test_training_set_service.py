from __future__ import annotations

from iread_ai.devtools.training_review_catalog import TRAINING_REVIEW_CATALOG
from iread_ai.generation_models import (
    TrainingActivityRequest,
    TrainingSetRequest,
    TrainingTargetFeature,
)
from iread_ai.lexicon.contracts import LexiconItem, LexiconPaletteResponse
from iread_ai.training_set_service import (
    AREA_POOLS,
    generate_training_activity,
    generate_training_set,
    plan_training_types,
)


def _target(feature_code: str) -> TrainingTargetFeature:
    return TrainingTargetFeature(
        featureCode=feature_code,
        weaknessScore=0.8,
        confidence=0.9,
        evidenceCount=10,
    )


class _WordPaletteService:
    def build_palette(self, request):
        feature_codes = {feature.featureCode for feature in request.targetFeatures}
        surfaces = (
            ["까치", "꼬리", "꾸미", "끼니", "깨비"]
            if "GRAPHEME.ONSET.TENSE.ㄲ" in feature_codes
            else ["나무", "바다", "모자", "기차", "토끼"]
        )
        return LexiconPaletteResponse(
            requestId=request.requestId,
            databaseVersion="test",
            analyzerVersion="test",
            items=[
                LexiconItem(
                    formId=index,
                    headword=surface,
                    surface=surface,
                    pronunciation=surface,
                    partOfSpeech="명사",
                    definition="테스트 낱말",
                    storyTier="CORE",
                    semanticTags=[],
                    syllableCount=len(surface),
                    batchimCount=0,
                    batchimRatio=0,
                    pronunciationStatus="VERIFIED",
                    features={},
                    score=10,
                    reasons=["test"],
                )
                for index, surface in enumerate(surfaces, start=1)
            ],
        )


def test_curriculum_area_pools_cover_all_34_training_types() -> None:
    expected = {spec.training_type for spec in TRAINING_REVIEW_CATALOG}
    covered = {
        training_type
        for area, training_types in AREA_POOLS.items()
        if area != "AUTO"
        for training_type in training_types
    }

    assert covered == expected


def test_vowel_focus_is_planned_as_five_different_activity_types() -> None:
    request = TrainingSetRequest(
        requestId="vowel-plan",
        schemaVersion=1,
        curriculumArea="LETTER_SOUND",
        activityCount=5,
        difficulty=1,
        targetFeatures=[_target("GRAPHEME.VOWEL.BASIC.ㅏ")],
    )

    assert plan_training_types(request) == (
        "VOWEL_TRACE",
        "VOWEL_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION",
        "PHONEME_BLEND",
        "BASIC_SYLLABLE_BUILD",
    )


def test_generates_vowel_training_set_with_canonical_trace_sound() -> None:
    request = TrainingSetRequest(
        requestId="vowel-set",
        schemaVersion=1,
        curriculumArea="LETTER_SOUND",
        activityCount=5,
        difficulty=1,
        targetFeatures=[_target("GRAPHEME.VOWEL.BASIC.ㅏ")],
    )

    result = generate_training_set(request, provider=None)
    response = result.response

    assert response.curriculumArea == "LETTER_SOUND"
    assert response.focusFeatureCodes == ["GRAPHEME.VOWEL.BASIC.ㅏ"]
    assert len(response.activities) == 5
    assert len({activity.trainingType for activity in response.activities}) == 5
    trace = response.activities[0]
    assert trace.trainingType == "VOWEL_TRACE"
    assert trace.provider == "rule-db"
    assert trace.item == {
        "vowelType": "BASIC",
        "target": "ㅏ",
        "soundText": "ㅏ",
        "traceAssetKey": "vowel_0",
    }


def test_each_of_the_34_types_can_generate_one_reviewable_activity() -> None:
    for sequence, spec in enumerate(TRAINING_REVIEW_CATALOG, start=1):
        request = TrainingActivityRequest(
            requestId=f"activity-{spec.training_type}",
            schemaVersion=1,
            sequence=((sequence - 1) % 5) + 1,
            trainingType=spec.training_type,
            difficulty=3,
            targetFeatures=[],
            excludedFeatures=[],
        )

        response = generate_training_activity(request, provider=None)

        assert response.activity.trainingType == spec.training_type
        assert response.activity.templateId == spec.template_id
        assert response.activity.item
        assert response.activity.provider in {"rule-db", "curated-fallback"}


def test_multi_target_word_activity_uses_target_palettes_and_rule_database() -> None:
    request = TrainingActivityRequest(
        requestId="multi-target-word-reading",
        schemaVersion=1,
        sequence=1,
        trainingType="WORD_READING",
        difficulty=2,
        targetFeatures=[
            _target("GRAPHEME.ONSET.TENSE.ㄲ"),
            _target("WORD.SYLLABLE_COUNT.2"),
        ],
        excludedFeatures=[],
    )

    response = generate_training_activity(
        request,
        provider=None,
        lexicon_service=_WordPaletteService(),
    )

    assert response.activity.provider == "rule-db"
    assert response.activity.item["readingOrder"] == "SEQUENTIAL"
    assert len(response.activity.item["words"]) == 4
    assert response.activity.personalization.lexiconApplied is True
