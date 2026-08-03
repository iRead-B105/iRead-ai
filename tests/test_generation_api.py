from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

import iread_ai.app as app_module

client = TestClient(app_module.app)
AUTH_HEADERS = {"X-API-Key": "test-internal-key"}


def request_headers(request_id: str) -> dict[str, str]:
    return {**AUTH_HEADERS, "Idempotency-Key": request_id}


@pytest.fixture(autouse=True)
def configured_internal_key(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "settings",
        replace(app_module.settings, internal_api_key=AUTH_HEADERS["X-API-Key"]),
    )


def candidate_request(training_type: str) -> dict:
    return {
        "requestId": f"contract-{training_type}",
        "schemaVersion": 2,
        "trainingType": training_type,
        "count": 5,
        "difficulty": 2,
        "targetFeatures": [],
        "excludedFeatures": [],
        "additionalPrompt": "정확히 3개의 선택지를 생성한다.",
        "outputTemplate": {"type": training_type, "data": [{}]},
    }


def test_all_training_types_generate_five_candidates() -> None:
    training_types = [
        "VOWEL_TRACE", "CONSONANT_TRACE", "SYLLABLE_TRACE",
        "CONSONANT_SOUND_CHOICE", "VOWEL_SOUND_CHOICE",
        "CONSONANT_VOWEL_CLASSIFICATION", "SYLLABLE_INITIAL_CHOICE",
        "WORD_INITIAL_CHOICE", "SAME_INITIAL_WORD_CHOICE",
        "FINAL_CONSONANT_CHOICE", "WORD_FINAL_SOUND_CHOICE",
        "FINAL_CONSONANT_COMPARISON", "SIMILAR_SOUND_CHOICE",
        "PHONEME_BLEND", "SYLLABLE_BLEND", "BASIC_SYLLABLE_BUILD",
        "FINAL_SYLLABLE_BUILD", "DOUBLE_FINAL_BUILD",
        "FINAL_CONSONANT_DELETE", "SYLLABLE_DELETE", "SYLLABLE_REPLACE",
        "WORD_READING", "NONWORD_READING", "DIFFICULT_WORD_PREVIEW",
        "SENTENCE_READING", "SHORT_PASSAGE_READING", "SENTENCE_ASSEMBLY",
        "FILL_IN_THE_BLANK", "IMAGE_SENTENCE_MATCH", "SENTENCE_REPEAT",
        "WORD_CHAIN_READING", "PHRASE_READING",
        "REPEATED_SENTENCE_READING", "SHORT_STORY_READING",
    ]
    for training_type in training_types:
        request = candidate_request(training_type)
        response = client.post(
            "/api/v1/trainings/candidates",
            headers={
                **AUTH_HEADERS,
                "Idempotency-Key": request["requestId"],
            },
            json=request,
        )
        assert response.status_code == 200, (training_type, response.text)
        assert response.json()["type"] == training_type
        assert len(response.json()["data"]) == 5


def test_multiple_choice_contract_uses_three_choices() -> None:
    three_choice_types = [
        "CONSONANT_SOUND_CHOICE", "VOWEL_SOUND_CHOICE",
        "SYLLABLE_INITIAL_CHOICE", "WORD_INITIAL_CHOICE",
        "SAME_INITIAL_WORD_CHOICE", "FINAL_CONSONANT_CHOICE",
        "WORD_FINAL_SOUND_CHOICE", "FINAL_CONSONANT_COMPARISON",
    ]
    for training_type in three_choice_types:
        request = candidate_request(training_type)
        response = client.post(
            "/api/v1/trainings/candidates",
            headers=request_headers(request["requestId"]),
            json=request,
        )
        assert all(len(candidate["choices"]) == 3 for candidate in response.json()["data"])


def test_story_generation_returns_day_one_first_four_pages() -> None:
    response = client.post(
        "/api/v1/story/generate",
        headers=request_headers("story-1"),
        json={
            "requestId": "story-1",
            "storyId": 1,
            "studentId": 2001,
            "schemaVersion": 1,
            "currentProgress": 0,
            "storyTemplate": {
                "storyTemplateId": 1,
                "title": "별빛 숲의 친구",
                "context": "숲에서 친구를 만나는 이야기",
            },
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["nextProgress"] == 4
    assert body["completed"] is False
    assert len(body["lines"]) == 4
    assert [line["requiresBranchInput"] for line in body["lines"]] == [
        False, False, False, True
    ]


def test_story_first_branch_returns_five_pages_and_reflects_intent() -> None:
    response = client.post(
        "/api/v1/story/continue",
        headers=request_headers("story-continue-1"),
        json={
            "requestId": "story-continue-1",
            "storyId": 1,
            "studentId": 2001,
            "schemaVersion": 1,
            "currentProgress": 4,
            "storyTemplate": {
                "storyTemplateId": 1,
                "title": "별빛 숲의 친구",
                "context": "숲에서 친구를 만나는 이야기",
            },
            "currentStoryLineId": 5,
            "branchIntent": "오른쪽 길로 갈래",
            "history": [
                {
                    "storyLineId": index,
                    "content": f"이야기 {index}",
                    "requiresBranchInput": index == 4,
                }
                for index in range(1, 5)
            ],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["nextProgress"] == 9
    assert body["completed"] is False
    assert len(body["lines"]) == 5
    assert "오른쪽 길로 갈래" in body["lines"][0]["content"]
    assert body["lines"][-1]["requiresBranchInput"] is True


def test_story_second_branch_closes_day_without_completing_story() -> None:
    response = client.post(
        "/api/v1/story/continue",
        headers=request_headers("story-continue-2"),
        json={
            "requestId": "story-continue-2",
            "storyId": 1,
            "studentId": 2001,
            "schemaVersion": 1,
            "currentProgress": 9,
            "storyTemplate": {
                "storyTemplateId": 1,
                "title": "토끼와 거북이",
                "context": "달리기 경주 이야기",
            },
            "currentStoryLineId": 9,
            "branchIntent": "토끼가 이겨?",
            "history": [
                {
                    "storyLineId": index,
                    "content": f"이야기 {index}",
                    "requiresBranchInput": index in (4, 9),
                }
                for index in range(1, 10)
            ],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["nextProgress"] == 10
    assert body["completed"] is False
    assert len(body["lines"]) == 1
    assert "토끼가 이겨" in body["lines"][0]["content"]


def test_story_completes_only_on_page_one_hundred() -> None:
    response = client.post(
        "/api/v1/story/continue",
        headers=request_headers("story-final"),
        json={
            "requestId": "story-final",
            "storyId": 1,
            "studentId": 2001,
            "schemaVersion": 1,
            "currentProgress": 99,
            "storyTemplate": {
                "storyTemplateId": 1,
                "title": "별빛 숲의 친구",
                "context": "숲에서 친구를 만나는 이야기",
            },
            "currentStoryLineId": 99,
            "branchIntent": "모두 함께 집으로 돌아가",
            "history": [
                {
                    "storyLineId": index,
                    "content": f"이야기 {index}",
                    "requiresBranchInput": index % 10 in (4, 9),
                }
                for index in range(1, 100)
            ],
        },
    )
    body = response.json()
    assert body["nextProgress"] == 100
    assert body["completed"] is True


def test_image_generation_returns_retrievable_svg() -> None:
    response = client.post(
        "/api/v1/images/generate",
        headers=request_headers("image-1"),
        json={"requestId": "image-1", "prompt": "[STORY_CHARACTER] 별빛 숲의 토끼"},
    )
    assert response.status_code == 200
    image = client.get(response.json()["imageUrl"])
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/svg+xml")


def test_internal_generation_rejects_missing_api_key() -> None:
    response = client.post(
        "/api/v1/trainings/candidates",
        json=candidate_request("VOWEL_TRACE"),
    )

    assert response.status_code == 401
