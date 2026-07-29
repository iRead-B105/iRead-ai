from fastapi.testclient import TestClient

import iread_ai.app as app_module
from iread_ai.config import Settings
from iread_ai.models import (
    PronunciationAnalysisResponse,
    PronunciationWordResult,
)


class FakeProvider:
    def analyze(
        self,
        *,
        request_id: str,
        reference_text: str,
        audio: bytes,
        original_filename: str | None,
    ) -> PronunciationAnalysisResponse:
        assert reference_text == "아기는 사과를"
        assert audio == b"audio"
        return PronunciationAnalysisResponse(
            requestId=request_id,
            pronunciationAccuracyScore=90,
            confidence=0.95,
            analysisVersion="TEST_V1",
            words=[
                PronunciationWordResult(
                    resultIndex=0,
                    word="아기는",
                    accuracyScore=91,
                    errorType="None",
                    offsetMs=100,
                    durationMs=500,
                ),
                PronunciationWordResult(
                    resultIndex=1,
                    word="사과를",
                    accuracyScore=89,
                    errorType="None",
                    offsetMs=650,
                    durationMs=600,
                ),
            ],
        )


def test_analyze_endpoint_returns_word_array(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            internal_api_key="shared-key",
            azure_speech_key="azure-key",
            azure_speech_region="koreacentral",
            azure_speech_language="ko-KR",
            max_audio_bytes=1024,
        ),
    )
    monkeypatch.setattr(app_module, "provider", FakeProvider())
    client = TestClient(app_module.app)

    response = client.post(
        "/api/v1/speech/pronunciation/analyze",
        data={
            "requestId": "request-1",
            "expectedText": "아기는 사과를",
        },
        files={"audioFile": ("sentence.wav", b"audio", "audio/wav")},
        headers={
            "X-API-Key": "shared-key",
            "Idempotency-Key": "request-1",
        },
    )

    assert response.status_code == 200
    assert [word["word"] for word in response.json()["words"]] == [
        "아기는",
        "사과를",
    ]


def test_rejects_wrong_internal_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            internal_api_key="shared-key",
            azure_speech_key="azure-key",
            azure_speech_region="koreacentral",
            azure_speech_language="ko-KR",
            max_audio_bytes=1024,
        ),
    )
    client = TestClient(app_module.app)

    response = client.post(
        "/api/v1/speech/pronunciation/analyze",
        data={"requestId": "request-1", "expectedText": "사과"},
        files={"audioFile": ("word.wav", b"audio", "audio/wav")},
        headers={
            "X-API-Key": "wrong",
            "Idempotency-Key": "request-1",
        },
    )

    assert response.status_code == 401
