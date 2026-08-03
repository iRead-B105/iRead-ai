from fastapi.testclient import TestClient

from iread_ai.app import app

client = TestClient(app)


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
            headers={"Idempotency-Key": request["requestId"]},
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
        response = client.post(
            "/api/v1/trainings/candidates",
            json=candidate_request(training_type),
        )
        assert all(len(candidate["choices"]) == 3 for candidate in response.json()["data"])


def test_story_generation_returns_five_lines_and_branch_at_line_five() -> None:
    response = client.post(
        "/api/v1/story/generate",
        headers={
            "X-API-Key": "local-development-key",
            "Idempotency-Key": "story-1",
        },
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
    assert body["nextProgress"] == 50
    assert body["completed"] is False
    assert len(body["lines"]) == 5
    assert [line["requiresBranchInput"] for line in body["lines"]] == [
        False, False, False, False, True
    ]
    assert all(line["branchPrompt"] is None for line in body["lines"][:4])
    assert [
        option["optionNo"] for option in body["lines"][-1]["branchPrompt"]["options"]
    ] == [1, 2, 3]
    assert len({
        option["label"] for option in body["lines"][-1]["branchPrompt"]["options"]
    }) == 3


def test_story_continue_returns_final_five_lines() -> None:
    response = client.post(
        "/api/v1/story/continue",
        headers={
            "X-API-Key": "local-development-key",
            "Idempotency-Key": "story-continue-1",
        },
        json={
            "requestId": "story-continue-1",
            "storyId": 1,
            "studentId": 2001,
            "schemaVersion": 1,
            "currentProgress": 50,
            "storyTemplate": {
                "storyTemplateId": 1,
                "title": "별빛 숲의 친구",
                "context": "숲에서 친구를 만나는 이야기",
            },
            "currentStoryLineId": 5,
            "branchIntent": "오른쪽 길로 갈래",
            "history": [],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["nextProgress"] == 100
    assert body["completed"] is True
    assert len(body["lines"]) == 5
    assert all(line["branchPrompt"] is None for line in body["lines"])


def test_story_generation_requires_internal_api_key() -> None:
    request = {
        "requestId": "story-auth-required",
        "storyId": 1,
        "studentId": 2001,
        "schemaVersion": 1,
        "currentProgress": 0,
        "storyTemplate": {
            "storyTemplateId": 1,
            "title": "별빛 숲의 친구",
            "context": "숲에서 친구를 만나는 이야기",
        },
    }
    response = client.post(
        "/api/v1/story/generate",
        headers={"Idempotency-Key": request["requestId"]},
        json=request,
    )
    assert response.status_code == 401


def test_story_generation_replays_same_idempotent_request() -> None:
    request = {
        "requestId": "story-replay",
        "storyId": 2,
        "studentId": 2001,
        "schemaVersion": 1,
        "currentProgress": 0,
        "storyTemplate": {
            "storyTemplateId": 2,
            "title": "구름 우체국",
            "context": "구름 위에서 편지를 전하는 이야기",
        },
    }
    headers = {
        "X-API-Key": "local-development-key",
        "Idempotency-Key": request["requestId"],
    }
    first = client.post("/api/v1/story/generate", headers=headers, json=request)
    second = client.post("/api/v1/story/generate", headers=headers, json=request)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.json() == first.json()


def test_story_generation_rejects_mismatched_idempotency_key() -> None:
    request = {
        "requestId": "story-key-mismatch",
        "storyId": 3,
        "studentId": 2001,
        "schemaVersion": 1,
        "currentProgress": 0,
        "storyTemplate": {
            "storyTemplateId": 1,
            "title": "별빛 숲의 친구",
            "context": "숲에서 친구를 만나는 이야기",
        },
    }
    response = client.post(
        "/api/v1/story/generate",
        headers={
            "X-API-Key": "local-development-key",
            "Idempotency-Key": "different-key",
        },
        json=request,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "IDEMPOTENCY_KEY_MISMATCH"


def test_story_generation_rejects_changed_body_for_same_key() -> None:
    request = {
        "requestId": "story-conflict",
        "storyId": 4,
        "studentId": 2001,
        "schemaVersion": 1,
        "currentProgress": 0,
        "storyTemplate": {
            "storyTemplateId": 1,
            "title": "별빛 숲의 친구",
            "context": "숲에서 친구를 만나는 이야기",
        },
    }
    headers = {
        "X-API-Key": "local-development-key",
        "Idempotency-Key": request["requestId"],
    }
    first = client.post("/api/v1/story/generate", headers=headers, json=request)
    changed = {**request, "storyId": 5}
    second = client.post("/api/v1/story/generate", headers=headers, json=changed)
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["code"] == "IDEMPOTENCY_CONFLICT"


def test_image_generation_returns_retrievable_svg() -> None:
    response = client.post(
        "/api/v1/images/generate",
        json={
            "requestId": "image-1",
            "prompt": "[STORY_CHARACTER] " + "한글 장면 설명" * 100,
        },
    )
    assert response.status_code == 200
    image = client.get(response.json()["imageUrl"])
    assert image.status_code == 200
    assert image.headers["content-type"].startswith("image/svg+xml")
    assert len(response.json()["imageUrl"]) < 100
