from __future__ import annotations

from pathlib import Path

import pytest

from iread_ai.devtools.training_review_catalog import (
    TRAINING_REVIEW_CATALOG,
    output_template,
)
from iread_ai.generation_models import TrainingCandidateRequest, TrainingTargetFeature
from iread_ai.personalization.hangul import decompose_text
from iread_ai.training_bank import (
    RULE_BASED_TYPES,
    RuleBasedBasicTrainingGenerator,
    SQLiteLearningUnitRepository,
)


def _request(training_type: str, feature_code: str) -> TrainingCandidateRequest:
    return TrainingCandidateRequest(
        requestId=f"bank-{training_type}-{feature_code}",
        schemaVersion=2,
        trainingType=training_type,
        count=5,
        difficulty=3,
        targetFeatures=[
            TrainingTargetFeature(
                featureCode=feature_code,
                weaknessScore=0.8,
                confidence=0.9,
                evidenceCount=10,
            )
        ],
        excludedFeatures=[],
        additionalPrompt="",
        outputTemplate={
            "type": training_type,
            "data": [
                {
                    "audioText": "<string>",
                    "choices": ["<string>"],
                    "answerIndex": "<integer>",
                }
            ],
        },
    )


def _generator(path: Path) -> RuleBasedBasicTrainingGenerator:
    return RuleBasedBasicTrainingGenerator(SQLiteLearningUnitRepository(path))


def test_builds_versioned_sqlite_bank_with_verified_atoms(tmp_path: Path) -> None:
    repository = SQLiteLearningUnitRepository(tmp_path / "bank.sqlite3")

    counts = repository.counts()
    units = repository.find_all_active()

    assert counts["units"] == 150
    assert counts["features"] >= 200
    assert counts["confusions"] == 38
    assert len(units) == 150
    assert {unit.unit_type for unit in units} == {
        "CONSONANT",
        "VOWEL",
        "SYLLABLE",
        "WORD",
    }
    assert any(
        unit.surface == "ㄱ" and "GRAPHEME.ONSET.BASIC.ㄱ" in unit.feature_codes for unit in units
    )
    assert any(
        unit.surface == "산" and "GRAPHEME.CODA.SIMPLE.ㄴ" in unit.feature_codes for unit in units
    )
    supported_trace_keys = {
        "consonant_0",
        "consonant_1",
        "consonant_2",
        "consonant_3",
        "consonant_4",
        "vowel_0",
        "vowel_1",
        "vowel_2",
        "vowel_3",
        "vowel_4",
        "syllable_0",
    }
    assert {unit.trace_asset_key for unit in units if unit.trace_asset_key} == supported_trace_keys


def test_generates_five_distinct_consonant_questions_for_profile_target(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "consonant.sqlite3").generate(
        _request("CONSONANT_SOUND_CHOICE", "GRAPHEME.ONSET.BASIC.ㄱ")
    )

    assert response is not None
    assert len(response.data) == 5
    assert len({repr(candidate) for candidate in response.data}) == 5
    for candidate in response.data:
        assert candidate["audioText"] == "ㄱ"
        assert len(candidate["choices"]) == 3
        assert candidate["choices"][candidate["answerIndex"]] == "ㄱ"


def test_generates_word_final_questions_without_calling_an_llm(tmp_path: Path) -> None:
    response = _generator(tmp_path / "word-final.sqlite3").generate(
        _request("WORD_FINAL_SOUND_CHOICE", "GRAPHEME.CODA.SIMPLE.ㄴ")
    )

    assert response is not None
    assert len(response.data) == 5
    for candidate in response.data:
        assert decompose_text(candidate["audioText"])[-1].coda == "ㄴ"
        assert candidate["choices"][candidate["answerIndex"]] == "ㄴ"


def test_keeps_the_requested_vowel_in_every_trace_candidate(tmp_path: Path) -> None:
    response = _generator(tmp_path / "vowel-trace.sqlite3").generate(
        _request("VOWEL_TRACE", "GRAPHEME.VOWEL.BASIC.ㅏ")
    )

    assert response is not None
    assert {candidate["target"] for candidate in response.data} == {"ㅏ"}
    assert {candidate["traceAssetKey"] for candidate in response.data} == {"vowel_0"}
    assert len({candidate["soundText"] for candidate in response.data}) == 5


def test_keeps_the_requested_grapheme_in_every_classification_candidate(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "classification.sqlite3").generate(
        _request("CONSONANT_VOWEL_CLASSIFICATION", "GRAPHEME.ONSET.BASIC.ㄱ")
    )

    assert response is not None
    assert len({repr(candidate) for candidate in response.data}) == 5
    for candidate in response.data:
        assert "ㄱ" in candidate["audioText"]
        assert candidate["choices"][candidate["answerIndex"]] == "CONSONANT"


def test_delegates_sentence_types_and_uncovered_targets(tmp_path: Path) -> None:
    generator = _generator(tmp_path / "delegation.sqlite3")

    assert generator.generate(_request("SENTENCE_READING", "SENTENCE.SIMPLE")) is None
    assert (
        generator.generate(_request("WORD_FINAL_SOUND_CHOICE", "GRAPHEME.CODA.COMPLEX.ㄳ")) is None
    )


@pytest.mark.parametrize(
    "spec",
    [spec for spec in TRAINING_REVIEW_CATALOG if spec.training_type in RULE_BASED_TYPES],
    ids=lambda spec: spec.training_type,
)
def test_all_rule_based_types_generate_five_semantically_valid_candidates(
    tmp_path: Path,
    spec,
) -> None:
    request = TrainingCandidateRequest(
        requestId=f"semantic-{spec.training_type}",
        schemaVersion=2,
        trainingType=spec.training_type,
        count=5,
        difficulty=3,
        targetFeatures=[
            TrainingTargetFeature(
                featureCode=spec.suggested_feature,
                weaknessScore=0.8,
                confidence=0.9,
                evidenceCount=10,
            )
        ],
        excludedFeatures=[],
        additionalPrompt="",
        outputTemplate=output_template(spec.training_type),
    )

    response = _generator(tmp_path / f"{spec.training_type}.sqlite3").generate(request)

    assert response is not None
    assert response.type == spec.training_type
    assert len(response.data) == 5
    assert len({repr(candidate) for candidate in response.data}) == 5
    for candidate in response.data:
        _assert_candidate_semantics(spec.training_type, candidate)


def _assert_candidate_semantics(training_type: str, candidate: dict) -> None:
    if "choices" in candidate and "answerIndex" in candidate:
        answer_index = candidate["answerIndex"]
        assert 0 <= answer_index < len(candidate["choices"])

    if training_type in {"SYLLABLE_INITIAL_CHOICE", "WORD_INITIAL_CHOICE"}:
        expected = decompose_text(candidate["audioText"])[0].onset
        assert candidate["choices"][candidate["answerIndex"]] == expected
    elif training_type == "SAME_INITIAL_WORD_CHOICE":
        target_onset = decompose_text(candidate["targetAudioText"])[0].onset
        selected = candidate["choices"][candidate["answerIndex"]]["text"]
        assert decompose_text(selected)[0].onset == target_onset
    elif training_type in {"FINAL_CONSONANT_CHOICE", "WORD_FINAL_SOUND_CHOICE"}:
        expected = decompose_text(candidate["audioText"])[-1].coda
        assert candidate["choices"][candidate["answerIndex"]] == expected
    elif training_type == "FINAL_CONSONANT_COMPARISON":
        assert candidate["choices"][candidate["answerIndex"]] == candidate["audioText"]
    elif training_type == "SIMILAR_SOUND_CHOICE":
        expected = decompose_text(candidate["audioText"])[0].onset
        assert candidate["choices"][candidate["answerIndex"]] == expected
    elif training_type == "PHONEME_BLEND":
        selected = [candidate["cards"][index] for index in candidate["answerOrder"]]
        result = decompose_text(candidate["result"])[0]
        expected = [result.onset, result.nucleus, *([result.coda] if result.coda else [])]
        assert selected == expected == candidate["audioParts"]
    elif training_type == "SYLLABLE_BLEND":
        selected = [candidate["cards"][index] for index in candidate["answerOrder"]]
        assert "".join(selected) == candidate["result"]
    elif training_type in {
        "BASIC_SYLLABLE_BUILD",
        "FINAL_SYLLABLE_BUILD",
        "DOUBLE_FINAL_BUILD",
    }:
        result = decompose_text(candidate["result"])[0]
        assert candidate["initialChoices"][candidate["initialAnswerIndex"]] == result.onset
        assert candidate["medialChoices"][candidate["medialAnswerIndex"]] == result.nucleus
        if training_type != "BASIC_SYLLABLE_BUILD":
            assert candidate["finalChoices"][candidate["finalAnswerIndex"]] == result.coda
    elif training_type == "FINAL_CONSONANT_DELETE":
        assert not decompose_text(candidate["result"])[0].coda
        assert candidate["removableUnits"][candidate["answerIndex"]]
    elif training_type == "SYLLABLE_DELETE":
        syllables = candidate["syllables"]
        expected = "".join(
            value for index, value in enumerate(syllables) if index != candidate["deleteIndex"]
        )
        assert candidate["result"] == expected
    elif training_type == "SYLLABLE_REPLACE":
        replacement = candidate["choices"][candidate["answerIndex"]]
        source = list(candidate["source"])
        source[candidate["replaceIndex"]] = replacement
        assert "".join(source) == candidate["result"]
    elif training_type == "WORD_READING":
        assert candidate["readingOrder"] == "SEQUENTIAL"
        assert len(candidate["words"]) == 3
    elif training_type == "NONWORD_READING":
        assert [item["isNonword"] for item in candidate["words"]] == [False, True]
        assert candidate["words"][0]["text"] != candidate["words"][1]["text"]
    elif training_type == "WORD_CHAIN_READING":
        assert candidate["requiredOrder"] == "SEQUENTIAL"
        assert len(candidate["words"]) == 4
