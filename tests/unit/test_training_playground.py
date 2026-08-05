from iread_ai.devtools.curriculum_samples import curriculum_sample
from iread_ai.devtools.training_playground import (
    SERVICE_TRAINING_CATALOG,
    build_candidate_request,
    build_target_features,
    correct_choice,
    expected_text,
    service_training_spec_by_id,
)


def test_playground_exposes_the_31_backend_training_templates() -> None:
    assert len(SERVICE_TRAINING_CATALOG) == 31
    assert {spec.template_id for spec in SERVICE_TRAINING_CATALOG}.isdisjoint({6, 14, 24})
    assert service_training_spec_by_id(34).name == "짧은 이야기 읽기"


def test_candidate_request_uses_at_most_two_profile_targets() -> None:
    sample = curriculum_sample("자모 읽기가 어려운 학생")
    targets = build_target_features(
        sample["featureProfiles"],
        [
            "GRAPHEME.VOWEL.BASIC.ㅏ",
            "GRAPHEME.ONSET.BASIC.ㄱ",
            "SENTENCE.SIMPLE",
        ],
    )
    request = build_candidate_request(
        request_id="playground-test",
        training_type="VOWEL_SOUND_CHOICE",
        difficulty=1,
        target_features=targets,
    )

    assert len(request["targetFeatures"]) == 2
    assert request["count"] == 5
    assert request["outputTemplate"]["type"] == "VOWEL_SOUND_CHOICE"


def test_playground_resolves_choice_and_text_answers() -> None:
    choice = {"choices": ["ㄱ", "ㄴ", "ㄷ"], "answerIndex": 1}
    assembly = {
        "cards": ["가요.", "아이가", "학교에"],
        "answerOrder": [1, 2, 0],
        "completedSentence": "아이가 학교에 가요.",
    }

    assert correct_choice(choice) == "ㄴ"
    assert expected_text("SENTENCE_ASSEMBLY", assembly) == "아이가 학교에 가요."
