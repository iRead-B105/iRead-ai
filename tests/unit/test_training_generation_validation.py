from __future__ import annotations

import pytest

from iread_ai.generation_models import (
    TrainingCandidateRequest,
    TrainingCandidateResponse,
    TrainingTargetFeature,
)
from iread_ai.generation_service import (
    _normalize_mechanical_fields,
    _validate_candidate_uniqueness,
    _validate_final_sound_choices,
    _validate_hybrid_semantics,
    enrich_training_request_with_lexicon,
    generate_training,
)
from iread_ai.lexicon.contracts import LexiconItem, LexiconPaletteResponse
from iread_ai.training_language_quality import validate_complete_korean_sentence


def _request(training_type: str) -> TrainingCandidateRequest:
    return TrainingCandidateRequest(
        requestId=f"validation-{training_type}",
        schemaVersion=2,
        trainingType=training_type,
        count=5,
        difficulty=3,
        targetFeatures=[],
        excludedFeatures=[],
        outputTemplate={"type": training_type, "data": [{}]},
    )


def test_final_consonant_choice_rejects_sentence_audio_and_non_representative_choices() -> None:
    request = _request("FINAL_CONSONANT_CHOICE").model_copy(update={"count": 1})
    response = TrainingCandidateResponse(
        type="FINAL_CONSONANT_CHOICE",
        data=[
            {
                "audioText": "가방에 책이 없다.",
                "choices": ["ㅊ", "ㅅ", "ㄷ"],
                "answerIndex": 0,
            }
        ],
    )

    with pytest.raises(ValueError):
        _validate_final_sound_choices(request, response)


def test_final_sound_choice_uses_standard_spoken_end_sound() -> None:
    request = _request("WORD_FINAL_SOUND_CHOICE").model_copy(update={"count": 2})
    response = TrainingCandidateResponse(
        type="WORD_FINAL_SOUND_CHOICE",
        data=[
            {"audioText": "꽃", "choices": ["ㄱ", "ㄷ", "ㅂ"], "answerIndex": 1},
            {"audioText": "밖", "choices": ["ㄴ", "ㄹ", "ㄱ"], "answerIndex": 2},
        ],
    )

    _validate_final_sound_choices(request, response)


def test_word_final_sound_choice_rejects_duplicate_words() -> None:
    request = _request("WORD_FINAL_SOUND_CHOICE").model_copy(update={"count": 2})
    response = TrainingCandidateResponse(
        type="WORD_FINAL_SOUND_CHOICE",
        data=[
            {"audioText": "하늘", "choices": ["ㄹ", "ㄱ", "ㄴ"], "answerIndex": 0},
            {"audioText": "하늘", "choices": ["ㅁ", "ㄹ", "ㅇ"], "answerIndex": 1},
        ],
    )

    with pytest.raises(ValueError):
        _validate_final_sound_choices(request, response)


class _LexiconService:
    def build_palette(self, request):
        feature = request.targetFeatures[0].featureCode if request.targetFeatures else "default"
        surfaces = {
            "GRAPHEME.ONSET.TENSE.ㄲ": ["꼬리", "까치", "끼니", "꾸러기"],
            "PHONOLOGY.NASALIZATION": ["국물", "앞니", "꽃망울", "먹는"],
            "default": ["토끼", "나무", "바다", "모자"],
        }.get(feature, ["토끼", "나무", "바다", "모자"])
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
                    definition="동물",
                    storyTier="CORE",
                    semanticTags=["동물"],
                    syllableCount=2,
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


class _UnsafeWordLexiconService(_LexiconService):
    def build_palette(self, request):
        response = super().build_palette(request)
        response.items.insert(
            0,
            LexiconItem(
                formId=999,
                headword="새끼",
                surface="새끼",
                pronunciation="새끼",
                partOfSpeech="명사",
                definition="어린 동물",
                storyTier="CORE",
                semanticTags=["동물"],
                syllableCount=2,
                batchimCount=1,
                batchimRatio=0.5,
                pronunciationStatus="VERIFIED",
                features={},
                score=100,
                reasons=["test"],
            ),
        )
        return response


def test_candidate_request_is_enriched_from_the_server_lexicon() -> None:
    request = _request("SENTENCE_READING")

    enriched = enrich_training_request_with_lexicon(request, _LexiconService())

    assert request.recommendedWords == []
    assert enriched.recommendedWords == ["토끼", "나무", "바다", "모자"]


def test_rule_based_candidate_request_does_not_query_the_lexicon() -> None:
    request = _request("VOWEL_TRACE")

    enriched = enrich_training_request_with_lexicon(request, _LexiconService())

    assert enriched is request


def test_word_reading_builds_a_separate_dictionary_palette_for_each_target() -> None:
    request = _request("WORD_READING").model_copy(
        update={
            "targetFeatures": [
                TrainingTargetFeature(
                    featureCode="GRAPHEME.ONSET.TENSE.ㄲ",
                    weaknessScore=0.9,
                    confidence=0.8,
                    evidenceCount=10,
                ),
                TrainingTargetFeature(
                    featureCode="PHONOLOGY.NASALIZATION",
                    weaknessScore=0.8,
                    confidence=0.9,
                    evidenceCount=8,
                ),
            ]
        }
    )

    enriched = enrich_training_request_with_lexicon(request, _LexiconService())

    assert enriched.recommendedWordsByFeature == {
        "GRAPHEME.ONSET.TENSE.ㄲ": ["꼬리", "까치", "끼니", "꾸러기"],
        "PHONOLOGY.NASALIZATION": ["국물", "앞니", "꽃망울", "먹는"],
    }


def test_word_length_target_is_translated_to_lexicon_syllable_bounds() -> None:
    class RecordingLexiconService(_LexiconService):
        def __init__(self) -> None:
            self.requests = []

        def build_palette(self, request):
            self.requests.append(request)
            return super().build_palette(request)

    service = RecordingLexiconService()
    request = _request("WORD_READING").model_copy(
        update={
            "targetFeatures": [
                TrainingTargetFeature(
                    featureCode="WORD.SYLLABLE_COUNT.2",
                    weaknessScore=0.9,
                    confidence=0.8,
                    evidenceCount=10,
                )
            ]
        }
    )

    enrich_training_request_with_lexicon(request, service)

    assert service.requests[0].minSyllables == 2
    assert service.requests[0].maxSyllables == 2
    assert service.requests[0].requireTarget is False


def test_child_training_palette_excludes_contextually_unsafe_words() -> None:
    request = _request("WORD_READING").model_copy(
        update={
            "targetFeatures": [
                TrainingTargetFeature(
                    featureCode="GRAPHEME.ONSET.TENSE.ㄲ",
                    weaknessScore=0.9,
                    confidence=0.8,
                    evidenceCount=10,
                )
            ]
        }
    )

    enriched = enrich_training_request_with_lexicon(request, _UnsafeWordLexiconService())

    assert "새끼" not in enriched.recommendedWords
    assert "새끼" not in enriched.recommendedWordsByFeature["GRAPHEME.ONSET.TENSE.ㄲ"]


def test_sentence_assembly_rejects_cards_that_are_already_ordered() -> None:
    item = {
        "cards": ["민수는", "공원에서", "놀아요."],
        "answerOrder": [0, 1, 2],
        "completedSentence": "민수는 공원에서 놀아요.",
    }
    response = TrainingCandidateResponse(
        type="SENTENCE_ASSEMBLY",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="not shuffled"):
        _validate_hybrid_semantics(_request("SENTENCE_ASSEMBLY"), response)


def test_sentence_assembly_accepts_shuffled_reconstructable_cards() -> None:
    item = {
        "cards": ["놀아요.", "민수는", "공원에서"],
        "answerOrder": [1, 2, 0],
        "completedSentence": "민수는 공원에서 놀아요.",
    }
    response = TrainingCandidateResponse(
        type="SENTENCE_ASSEMBLY",
        data=[dict(item) for _ in range(5)],
    )

    _validate_hybrid_semantics(_request("SENTENCE_ASSEMBLY"), response)


def test_sentence_assembly_is_deterministically_shuffled_before_validation() -> None:
    item = {
        "cards": ["민수는", "공원에서", "놀아요."],
        "answerOrder": [0, 1, 2],
        "completedSentence": "민수는 공원에서 놀아요.",
    }
    response = TrainingCandidateResponse(
        type="SENTENCE_ASSEMBLY",
        data=[dict(item) for _ in range(5)],
    )

    request = _request("SENTENCE_ASSEMBLY")
    _normalize_mechanical_fields(request, response)
    _validate_hybrid_semantics(request, response)

    assert response.data[0]["cards"] == ["놀아요.", "민수는", "공원에서"]
    assert response.data[0]["answerOrder"] == [1, 2, 0]


def test_trace_sound_text_is_always_normalized_to_the_displayed_target() -> None:
    response = TrainingCandidateResponse(
        type="SYLLABLE_TRACE",
        data=[
            {
                "syllableType": "WITH_FINAL",
                "target": "읽",
                "soundText": "나는 짧은 글을 읽어요.",
                "traceAssetKey": "syllable_read",
            }
        ],
    )

    _normalize_mechanical_fields(_request("SYLLABLE_TRACE"), response)

    assert response.data[0]["soundText"] == "읽"


def test_fill_blank_rejects_incorrect_korean_particle_agreement() -> None:
    item = {
        "sentence": "민수는 {{blank}}를 들고 왔어요.",
        "inputType": "CHOICE",
        "choices": ["연필", "책상", "창문"],
        "answerIndex": 0,
        "acceptedAnswers": ["연필"],
        "completedSentence": "민수는 연필를 들고 왔어요.",
    }
    response = TrainingCandidateResponse(
        type="FILL_IN_THE_BLANK",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="Korean particle"):
        _validate_hybrid_semantics(_request("FILL_IN_THE_BLANK"), response)


def test_fill_blank_accepts_answers_with_matching_korean_particle() -> None:
    item = {
        "sentence": "민수는 {{blank}}을 들고 왔어요.",
        "inputType": "CHOICE",
        "choices": ["연필", "책상", "창문"],
        "answerIndex": 0,
        "acceptedAnswers": ["연필"],
        "completedSentence": "민수는 연필을 들고 왔어요.",
    }
    response = TrainingCandidateResponse(
        type="FILL_IN_THE_BLANK",
        data=[dict(item) for _ in range(5)],
    )

    _validate_hybrid_semantics(_request("FILL_IN_THE_BLANK"), response)


def test_fill_blank_rejects_a_missing_particle_after_the_blank() -> None:
    item = {
        "sentence": "민지는 {{blank}} 접고 창가로 갔어요.",
        "inputType": "CHOICE",
        "choices": ["종이", "편지", "수건"],
        "answerIndex": 0,
        "acceptedAnswers": ["종이"],
        "completedSentence": "민지는 종이 접고 창가로 갔어요.",
    }
    response = TrainingCandidateResponse(
        type="FILL_IN_THE_BLANK",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="followed by a Korean particle"):
        _validate_hybrid_semantics(_request("FILL_IN_THE_BLANK"), response)


def test_korean_sentence_validator_rejects_particle_and_name_suffix_errors() -> None:
    with pytest.raises(ValueError, match="particle agreement"):
        validate_complete_korean_sentence("아이가 연필를 들고 가요.")
    with pytest.raises(ValueError, match="name particle"):
        validate_complete_korean_sentence("소라이는 숲으로 갔어요.")
    with pytest.raises(ValueError, match="subject particle for a desired object"):
        validate_complete_korean_sentence("민수는 따뜻한 국물이 먹고 싶었어요.")


def test_image_sentence_match_requires_the_answer_to_match_the_image_prompt() -> None:
    item = {
        "imagePrompt": "숲에서 토끼가 사과를 먹는 장면",
        "choices": [
            "고양이가 방에서 자요.",
            "토끼가 숲에서 사과를 먹어요.",
            "비행기가 하늘을 날아요.",
        ],
        "answerIndex": 0,
    }
    response = TrainingCandidateResponse(
        type="IMAGE_SENTENCE_MATCH",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="uniquely support"):
        _validate_hybrid_semantics(_request("IMAGE_SENTENCE_MATCH"), response)


def test_provider_batch_rejects_duplicate_candidates() -> None:
    item = {"sentence": "토끼가 숲길을 걸어요.", "tokens": ["토끼가", "숲길을", "걸어요."]}
    response = TrainingCandidateResponse(
        type="SENTENCE_READING",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="must not be duplicated"):
        _validate_candidate_uniqueness(response)


def test_short_story_character_line_must_be_direct_dialogue() -> None:
    item = {
        "title": "숲속 문",
        "sentences": [
            {
                "speaker": "NARRATOR",
                "text": "강아지가 숲길에서 작은 문을 찾았어요.",
                "emotion": "CALM",
            },
            {
                "speaker": "CHARACTER",
                "text": "강아지가 문 안을 궁금해했어요.",
                "emotion": "SURPRISED",
            },
            {
                "speaker": "NARRATOR",
                "text": "문을 열자 밝은 별이 나왔어요.",
                "emotion": "HAPPY",
            },
        ],
    }
    response = TrainingCandidateResponse(
        type="SHORT_STORY_READING",
        data=[dict(item) for _ in range(5)],
    )

    with pytest.raises(ValueError, match="direct quoted dialogue"):
        _validate_hybrid_semantics(_request("SHORT_STORY_READING"), response)


class _QueuedProvider:
    model = "gpt-test"

    def __init__(
        self,
        documents: list[dict],
        *,
        provider_name: str = "gms",
    ) -> None:
        self.documents = documents
        self.provider_name = provider_name
        self.calls: list[dict] = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.documents.pop(0)


class _LexiconValidator:
    def __init__(self, unknown: list[str]) -> None:
        self.unknown = unknown
        self.texts: list[str] = []

    def unknown_content_words(self, texts: list[str]) -> list[str]:
        self.texts.extend(texts)
        return [word for word in self.unknown if any(word in text for text in texts)]


def test_short_passage_uses_a_second_llm_call_for_semantic_review() -> None:
    characters = ["토끼", "거북이", "다람쥐", "강아지", "고양이"]
    draft = {
        "type": "SHORT_PASSAGE_READING",
        "data": [
            {
                "sentences": [
                    f"{character}가 작은 문을 열었어요.",
                    "안에서 밝은 별빛이 나왔어요.",
                ]
            }
            for character in characters
        ],
    }
    reviewed = {
        "type": "SHORT_PASSAGE_READING",
        "data": [
            {
                "sentences": [
                    f"{character}가 작은 문을 열었어요.",
                    "그러자 안에서 밝은 별빛이 나왔어요.",
                ]
            }
            for character in characters
        ],
    }
    provider = _QueuedProvider([draft, reviewed], provider_name="openai")
    request = _request("SHORT_PASSAGE_READING").model_copy(
        update={
            "useLexicon": False,
            "outputTemplate": {"data": [{"sentences": ["<string>"]}]},
        }
    )

    result = generate_training(request, provider)

    assert result.provider == "openai:gpt-test"
    assert result.value.generationMetadata is not None
    assert result.value.generationMetadata.provider == "openai"
    assert len(provider.calls) == 2
    assert provider.calls[1]["schema_name"] == "iread_training_candidates_reviewed"
    assert result.value.data == reviewed["data"]


def test_local_validation_feedback_is_sent_on_the_single_retry() -> None:
    sentences = [
        "토끼가 숲길을 걸어요.",
        "거북이가 물가를 걸어요.",
        "다람쥐가 나뭇가지를 올라요.",
        "강아지가 공원을 달려요.",
        "고양이가 창가에 앉아요.",
    ]
    invalid = {
        "type": "SENTENCE_READING",
        "data": [{"sentence": sentence, "tokens": ["틀린", "토큰"]} for sentence in sentences],
    }
    valid = {
        "type": "SENTENCE_READING",
        "data": [
            {"sentence": sentence, "tokens": sentence.split()}
            for sentence in sentences
        ],
    }
    provider = _QueuedProvider([invalid, valid])
    request = _request("SENTENCE_READING").model_copy(
        update={
            "useLexicon": False,
            "outputTemplate": {
                "data": [{"sentence": "<string>", "tokens": ["<string>"]}]
            },
        }
    )

    result = generate_training(request, provider)

    assert result.value.data == valid["data"]
    assert len(provider.calls) == 2
    assert "previousValidationFailure" in provider.calls[1]["user_prompt"]


def test_sentence_generation_retries_when_registered_word_validation_finds_nonword() -> None:
    invalid = {
        "type": "SENTENCE_READING",
        "data": [
            {
                "sentence": f"토끼가 까나를 {verb}.",
                "tokens": ["토끼가", "까나를", f"{verb}."],
            }
            for verb in ["봐요", "들어요", "놓아요", "찾아요", "열어요"]
        ],
    }
    valid_sentences = [
        "토끼가 꽃을 봐요.",
        "거북이가 책을 들어요.",
        "아기가 공을 놓아요.",
        "친구가 길을 찾아요.",
        "엄마가 문을 열어요.",
    ]
    valid = {
        "type": "SENTENCE_READING",
        "data": [
            {"sentence": sentence, "tokens": sentence.split()}
            for sentence in valid_sentences
        ],
    }
    provider = _QueuedProvider([invalid, valid])
    lexicon = _LexiconValidator(["까나"])
    request = _request("SENTENCE_READING").model_copy(
        update={
            "useLexicon": False,
            "outputTemplate": {
                "data": [{"sentence": "<string>", "tokens": ["<string>"]}]
            },
        }
    )

    result = generate_training(request, provider, lexicon_service=lexicon)

    assert result.value.data == valid["data"]
    assert len(provider.calls) == 2
    assert "previousValidationFailure" in provider.calls[1]["user_prompt"]
    assert result.value.generationMetadata is not None
    assert result.value.generationMetadata.lexicalPolicy == "REAL_WORD_ONLY"


def test_word_level_generation_does_not_apply_registered_word_validation() -> None:
    request = _request("WORD_READING")
    lexicon = _LexiconValidator(["까나"])

    result = generate_training(request, provider=None, lexicon_service=lexicon)

    assert lexicon.texts == []
    assert result.value.generationMetadata is not None
    assert result.value.generationMetadata.lexicalPolicy == "PSEUDOWORD_ALLOWED"


def test_candidate_response_accepts_backend_question_count_of_three() -> None:
    # 백엔드는 훈련당 문항 수를 3으로 보낸다. 요청(count)과 응답(data 길이) 모두
    # 5 고정이 아니라 1~9 를 받아야 실제 생성 전환 시 계약 검증에서 거부되지 않는다.
    request = _request("WORD_READING").model_copy(update={"count": 3})
    assert request.count == 3

    response = TrainingCandidateResponse(
        type="WORD_READING",
        data=[{"word": "가"}, {"word": "나"}, {"word": "다"}],
    )
    assert len(response.data) == 3
