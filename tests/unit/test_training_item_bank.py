from __future__ import annotations

import re
from pathlib import Path

import pytest

from iread_ai.devtools.training_review_catalog import (
    TRAINING_REVIEW_CATALOG,
    output_template,
)
from iread_ai.generation_models import TrainingCandidateRequest, TrainingTargetFeature
from iread_ai.personalization.hangul import COMPLEX_CODA_PARTS, decompose_text
from iread_ai.training_bank import (
    RULE_BASED_TYPES,
    RuleBasedBasicTrainingGenerator,
    SQLiteLearningUnitRepository,
)
from iread_ai.training_bank.seed import unit_seeds
from iread_ai.training_final_sounds import representative_final_sound


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

    assert counts["units"] == 156
    assert counts["features"] >= 200
    assert counts["confusions"] == 38
    assert len(units) == 156
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
        "syllable_1",
        "syllable_2",
        "syllable_3",
        "syllable_4",
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
    assert len({candidate["audioText"] for candidate in response.data}) == 5
    for candidate in response.data:
        assert decompose_text(candidate["audioText"])[-1].coda == "ㄴ"
        assert candidate["choices"][candidate["answerIndex"]] == "ㄴ"


def test_final_consonant_choice_uses_distinct_syllables_and_spoken_final_sound(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "final-consonant.sqlite3").generate(
        _request("FINAL_CONSONANT_CHOICE", "GRAPHEME.CODA.SIMPLE.ㅊ")
    )

    assert response is not None
    assert len(response.data) == 5
    assert len({candidate["audioText"] for candidate in response.data}) == 5
    for candidate in response.data:
        assert len(candidate["audioText"]) == 1
        assert decompose_text(candidate["audioText"])[-1].coda == "ㅊ"
        assert set(candidate["choices"]) <= {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"}
        assert candidate["choices"][candidate["answerIndex"]] == "ㄷ"


def test_final_consonant_comparison_uses_audibly_distinct_choices(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "final-comparison.sqlite3").generate(
        _request("FINAL_CONSONANT_COMPARISON", "GRAPHEME.CODA.SIMPLE.ㄱ")
    )

    assert response is not None
    for candidate in response.data:
        choices = candidate["choices"]
        assert choices[candidate["answerIndex"]] == candidate["audioText"]
        assert len({representative_final_sound(choice) for choice in choices}) == 3
        bases = {(ord(choice) - 0xAC00) // 28 for choice in choices}
        assert len(bases) == 1


def test_keeps_the_requested_vowel_and_adds_distinct_review_trace_candidates(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "vowel-trace.sqlite3").generate(
        _request("VOWEL_TRACE", "GRAPHEME.VOWEL.BASIC.ㅏ")
    )

    assert response is not None
    assert response.data[0]["target"] == "ㅏ"
    assert len({candidate["target"] for candidate in response.data}) == 5
    assert all(candidate["soundText"] == candidate["target"] for candidate in response.data)


def test_consonant_trace_sound_text_contains_only_the_pronunciation_target(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "consonant-trace.sqlite3").generate(
        _request("CONSONANT_TRACE", "GRAPHEME.ONSET.BASIC.ㅁ")
    )

    assert response is not None
    assert response.data[0]["target"] == "ㅁ"
    assert len({candidate["target"] for candidate in response.data}) == 5
    assert all(candidate["soundText"] == candidate["target"] for candidate in response.data)


def test_incompatible_profile_feature_never_delegates_a_trace_item_to_llm(
    tmp_path: Path,
) -> None:
    response = _generator(tmp_path / "incompatible-vowel-trace.sqlite3").generate(
        _request("VOWEL_TRACE", "GRAPHEME.ONSET.TENSE.ㄲ")
    )

    assert response is not None
    assert len(response.data) == 5
    assert all(candidate["target"] in {"ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ"} for candidate in response.data)


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


def test_distributes_two_targets_across_five_rule_based_questions(tmp_path: Path) -> None:
    request = _request("CONSONANT_VOWEL_CLASSIFICATION", "GRAPHEME.ONSET.BASIC.ㄱ")
    request = request.model_copy(
        update={
            "targetFeatures": [
                *request.targetFeatures,
                TrainingTargetFeature(
                    featureCode="GRAPHEME.VOWEL.BASIC.ㅏ",
                    weaknessScore=0.7,
                    confidence=0.8,
                    evidenceCount=8,
                ),
            ]
        }
    )

    response = _generator(tmp_path / "multi-target.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    assert all("ㄱ" in candidate["audioText"] for candidate in response.data[:3])
    assert all("ㅏ" in candidate["audioText"] for candidate in response.data[3:])
    assert all(
        candidate["choices"][candidate["answerIndex"]] == "CONSONANT"
        for candidate in response.data[:3]
    )
    assert all(
        candidate["choices"][candidate["answerIndex"]] == "VOWEL"
        for candidate in response.data[3:]
    )


def test_multi_target_nonword_training_keeps_real_and_meaningless_items(
    tmp_path: Path,
) -> None:
    request = _request("NONWORD_READING", "GRAPHEME.ONSET.BASIC.ㄱ")
    request = request.model_copy(
        update={
            "targetFeatures": [
                *request.targetFeatures,
                TrainingTargetFeature(
                    featureCode="GRAPHEME.ONSET.BASIC.ㄴ",
                    weaknessScore=0.7,
                    confidence=0.8,
                    evidenceCount=8,
                ),
            ],
            "outputTemplate": {
                "type": "NONWORD_READING",
                "data": [{"words": [{"text": "<string>", "isNonword": "<boolean>"}]}],
            },
        }
    )

    response = _generator(tmp_path / "multi-target-nonword.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    for candidate in response.data:
        flags = {word["isNonword"] for word in candidate["words"]}
        assert flags == {False, True}


def test_word_reading_uses_target_specific_lexicon_words(tmp_path: Path) -> None:
    request = _request("WORD_READING", "GRAPHEME.ONSET.TENSE.ㄲ")
    request = request.model_copy(
        update={
            "targetFeatures": [
                *request.targetFeatures,
                TrainingTargetFeature(
                    featureCode="PHONOLOGY.NASALIZATION",
                    weaknessScore=0.7,
                    confidence=0.8,
                    evidenceCount=8,
                ),
            ],
            "recommendedWordsByFeature": {
                "GRAPHEME.ONSET.TENSE.ㄲ": ["꼬리", "까치", "끼니", "꾸러기", "깨소금"],
                "PHONOLOGY.NASALIZATION": ["국물", "앞니", "꽃망울", "먹는", "막내"],
            },
        }
    )

    response = _generator(tmp_path / "lexicon-word-reading.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    first_palette = set(request.recommendedWordsByFeature["GRAPHEME.ONSET.TENSE.ㄲ"])
    second_palette = set(request.recommendedWordsByFeature["PHONOLOGY.NASALIZATION"])
    for candidate in response.data:
        assert len(candidate["words"]) == 4
        assert len(set(candidate["words"]).intersection(first_palette)) == 2
        assert len(set(candidate["words"]).intersection(second_palette)) == 2


def test_word_reading_rejects_phrase_like_palette_entries(tmp_path: Path) -> None:
    request = _request("WORD_READING", "WORD.SYLLABLE_COUNT.2").model_copy(
        update={
            "recommendedWordsByFeature": {
                "WORD.SYLLABLE_COUNT.2": [
                    "책 읽는 아이",
                    "국물 맛",
                    "밥 먹기",
                    "읽는 책",
                ]
            }
        }
    )

    response = _generator(tmp_path / "phrase-rejection.sqlite3").generate(request)

    assert response is not None
    for candidate in response.data:
        assert all(" " not in word for word in candidate["words"])


def test_sparse_target_palette_is_supplemented_without_losing_the_target(tmp_path: Path) -> None:
    target_words = ["꼬리", "까치"]
    request = _request("WORD_READING", "GRAPHEME.ONSET.TENSE.ㄲ").model_copy(
        update={
            "recommendedWords": [
                *target_words,
                "나무",
                "바다",
                "모자",
                "기차",
                "토끼",
            ],
            "recommendedWordsByFeature": {
                "GRAPHEME.ONSET.TENSE.ㄲ": target_words,
            },
        }
    )

    response = _generator(tmp_path / "sparse-target-palette.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    for candidate in response.data:
        assert len(candidate["words"]) == 4
        assert set(candidate["words"]).intersection(target_words)


def test_word_length_target_is_applied_to_every_word_in_a_multi_target_item(
    tmp_path: Path,
) -> None:
    request = _request("WORD_READING", "GRAPHEME.ONSET.TENSE.ㄲ").model_copy(
        update={
            "targetFeatures": [
                *_request("WORD_READING", "GRAPHEME.ONSET.TENSE.ㄲ").targetFeatures,
                TrainingTargetFeature(
                    featureCode="WORD.SYLLABLE_COUNT.2",
                    weaknessScore=0.7,
                    confidence=0.8,
                    evidenceCount=8,
                ),
            ],
            "recommendedWords": [
                "토끼",
                "어깨",
                "꼬리",
                "새끼",
                "나무",
                "바다",
                "꽃나무",
            ],
            "recommendedWordsByFeature": {
                "GRAPHEME.ONSET.TENSE.ㄲ": [
                    "토끼",
                    "어깨",
                    "꼬리",
                    "새끼",
                    "꽃나무",
                ],
                "WORD.SYLLABLE_COUNT.2": ["나무", "바다", "토끼", "어깨"],
            },
        }
    )

    response = _generator(tmp_path / "multi-target-word-length.sqlite3").generate(request)

    assert response is not None
    for candidate in response.data:
        assert len(candidate["words"]) == 4
        assert all(len(word) == 2 for word in candidate["words"])
        tense_word_count = sum(
            "ㄲ" in [syllable.onset for syllable in decompose_text(word)]
            for word in candidate["words"]
        )
        assert tense_word_count == 2
        contrast_words = [
            word
            for word in candidate["words"]
            if "ㄲ" not in [syllable.onset for syllable in decompose_text(word)]
        ]
        curated_words = {
            seed.surface
            for seed in unit_seeds()
            if seed.unit_type == "WORD"
            and seed.familiarity >= 4
            and seed.difficulty <= request.difficulty
            and len(seed.surface) == 2
        }
        assert set(contrast_words).issubset(curated_words)


@pytest.mark.parametrize(
    ("training_type", "feature_code", "words"),
    [
        (
            "WORD_INITIAL_CHOICE",
            "GRAPHEME.ONSET.TENSE.ㄲ",
            ["꼬리", "까치", "끼니", "꾸러기", "깨소금", "꼬마"],
        ),
        (
            "SAME_INITIAL_WORD_CHOICE",
            "GRAPHEME.ONSET.TENSE.ㄲ",
            ["꼬리", "까치", "끼니", "꾸러기", "깨소금", "꼬마"],
        ),
        (
            "WORD_FINAL_SOUND_CHOICE",
            "GRAPHEME.CODA.SIMPLE.ㄴ",
            ["산", "문", "손", "눈", "우산", "기린"],
        ),
        (
            "SYLLABLE_BLEND",
            "WORD.SYLLABLE_COUNT.2",
            ["나무", "바다", "토끼", "모자", "사과", "나비"],
        ),
        (
            "SYLLABLE_DELETE",
            "WORD.SYLLABLE_COUNT.2",
            ["나무", "바다", "토끼", "모자", "사과", "나비"],
        ),
        (
            "NONWORD_READING",
            "WORD.SYLLABLE_COUNT.2",
            ["나무", "바다", "토끼", "모자", "사과", "나비"],
        ),
        (
            "WORD_CHAIN_READING",
            "WORD.SYLLABLE_COUNT.2",
            ["나무", "바다", "토끼", "모자", "사과", "나비"],
        ),
    ],
)
def test_word_training_types_use_verified_lexicon_units(
    tmp_path: Path,
    training_type: str,
    feature_code: str,
    words: list[str],
) -> None:
    request = _request(training_type, feature_code).model_copy(
        update={"recommendedWordsByFeature": {feature_code: words}}
    )

    response = _generator(tmp_path / f"lexicon-{training_type}.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    palette = set(words)
    for candidate in response.data:
        if training_type in {"WORD_INITIAL_CHOICE", "WORD_FINAL_SOUND_CHOICE"}:
            assert candidate["audioText"] in palette
        elif training_type == "SAME_INITIAL_WORD_CHOICE":
            assert candidate["targetAudioText"] in palette
            selected = candidate["choices"][candidate["answerIndex"]]["text"]
            assert selected in palette
            assert selected != candidate["targetAudioText"]
        elif training_type in {"SYLLABLE_BLEND", "SYLLABLE_DELETE"}:
            key = "result" if training_type == "SYLLABLE_BLEND" else "source"
            assert candidate[key] in palette
        elif training_type == "NONWORD_READING":
            real_words = [item["text"] for item in candidate["words"] if not item["isNonword"]]
            assert set(real_words).issubset(palette)
        elif training_type == "WORD_CHAIN_READING":
            assert set(candidate["words"]).issubset(palette)


def test_syllable_delete_distributes_delete_positions_across_the_set(
    tmp_path: Path,
) -> None:
    feature_code = "WORD.SYLLABLE_COUNT.2"
    request = _request("SYLLABLE_DELETE", feature_code).model_copy(
        update={
            "recommendedWordsByFeature": {
                feature_code: ["나무", "바다", "토끼", "모자", "사과", "나비"]
            }
        }
    )

    response = _generator(tmp_path / "syllable-delete-positions.sqlite3").generate(request)

    assert response is not None
    assert len({candidate["source"] for candidate in response.data}) == 5
    assert [candidate["deleteIndex"] for candidate in response.data] == [0, 1, 0, 1, 0]


def test_delegates_sentence_types_and_uncovered_targets(tmp_path: Path) -> None:
    generator = _generator(tmp_path / "delegation.sqlite3")

    assert generator.generate(_request("SENTENCE_READING", "SENTENCE.SIMPLE")) is None
    assert (
        generator.generate(_request("WORD_FINAL_SOUND_CHOICE", "GRAPHEME.CODA.COMPLEX.ㄳ")) is None
    )


def test_final_consonant_delete_uses_varied_familiar_syllables(tmp_path: Path) -> None:
    request = _request("FINAL_CONSONANT_DELETE", "SYLLABLE.CVC").model_copy(
        update={"outputTemplate": output_template("FINAL_CONSONANT_DELETE")}
    )

    response = _generator(tmp_path / "final-delete-cvc.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    assert len({candidate["source"] for candidate in response.data}) == 5
    assert len({candidate["result"] for candidate in response.data}) == 5
    for candidate in response.data:
        source = decompose_text(candidate["source"])[0]
        result = decompose_text(candidate["result"])[0]
        assert source.coda
        assert (source.onset, source.nucleus) == (result.onset, result.nucleus)
        assert not result.coda
        assert candidate["removableUnits"] == [
            source.onset,
            source.nucleus,
            source.coda,
        ]
        assert candidate["answerIndex"] == 2


def test_final_consonant_delete_honors_exact_coda_target(tmp_path: Path) -> None:
    request = _request(
        "FINAL_CONSONANT_DELETE",
        "GRAPHEME.CODA.SIMPLE.ㄱ",
    ).model_copy(update={"outputTemplate": output_template("FINAL_CONSONANT_DELETE")})

    response = _generator(tmp_path / "final-delete-coda.sqlite3").generate(request)

    assert response is not None
    assert len(response.data) == 5
    for candidate in response.data:
        source = decompose_text(candidate["source"])[0]
        assert source.coda == "ㄱ"
        assert candidate["removableUnits"][candidate["answerIndex"]] == "ㄱ"


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
        assert candidate["targetAudioText"] == candidate["result"]
        assert candidate["initialChoices"][candidate["initialAnswerIndex"]] == result.onset
        assert candidate["medialChoices"][candidate["medialAnswerIndex"]] == result.nucleus
        if training_type == "BASIC_SYLLABLE_BUILD":
            assert not result.coda
            assert "finalChoices" not in candidate
        elif training_type == "FINAL_SYLLABLE_BUILD":
            assert result.coda and result.coda not in COMPLEX_CODA_PARTS
            assert all(choice not in COMPLEX_CODA_PARTS for choice in candidate["finalChoices"])
            assert candidate["finalChoices"][candidate["finalAnswerIndex"]] == result.coda
        else:
            assert result.coda in COMPLEX_CODA_PARTS
            assert all(choice in COMPLEX_CODA_PARTS for choice in candidate["finalChoices"])
            assert candidate["finalChoices"][candidate["finalAnswerIndex"]] == result.coda
    elif training_type == "FINAL_CONSONANT_DELETE":
        source = decompose_text(candidate["source"])[0]
        result = decompose_text(candidate["result"])[0]
        assert (source.onset, source.nucleus) == (result.onset, result.nucleus)
        assert source.coda and not result.coda
        assert candidate["removableUnits"] == [
            source.onset,
            source.nucleus,
            source.coda,
        ]
        assert candidate["answerIndex"] == 2
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
        assert len(candidate["words"]) == 4
        assert all(re.fullmatch(r"[가-힣]+", word) for word in candidate["words"])
    elif training_type == "NONWORD_READING":
        assert [item["isNonword"] for item in candidate["words"]] == [False, True]
        assert candidate["words"][0]["text"] != candidate["words"][1]["text"]
    elif training_type == "WORD_CHAIN_READING":
        assert candidate["requiredOrder"] == "SEQUENTIAL"
        assert len(candidate["words"]) == 4
