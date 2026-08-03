from __future__ import annotations

import pytest

from iread_ai.generation_models import TrainingCandidateRequest, TrainingCandidateResponse
from iread_ai.generation_service import (
    _normalize_mechanical_fields,
    _validate_hybrid_semantics,
    enrich_training_request_with_lexicon,
)
from iread_ai.lexicon.contracts import LexiconItem, LexiconPaletteResponse


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


class _LexiconService:
    def build_palette(self, request):
        return LexiconPaletteResponse(
            requestId=request.requestId,
            databaseVersion="test",
            analyzerVersion="test",
            items=[
                LexiconItem(
                    formId=1,
                    headword="토끼",
                    surface="토끼",
                    pronunciation="토끼",
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
            ],
        )


def test_candidate_request_is_enriched_from_the_server_lexicon() -> None:
    request = _request("SENTENCE_READING")

    enriched = enrich_training_request_with_lexicon(request, _LexiconService())

    assert request.recommendedWords == []
    assert enriched.recommendedWords == ["토끼"]


def test_rule_based_candidate_request_does_not_query_the_lexicon() -> None:
    request = _request("VOWEL_TRACE")

    enriched = enrich_training_request_with_lexicon(request, _LexiconService())

    assert enriched is request


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
