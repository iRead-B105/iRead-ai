from __future__ import annotations

from typing import Any

import httpx
import pytest

import iread_ai.adapters.generation.gemini_responses as gemini_module
from iread_ai.adapters.generation.gemini_responses import GeminiResponsesClient


@pytest.mark.asyncio
async def test_gemini_35_flash_lite_uses_supported_generate_content_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            captured.update(url=url, headers=headers, payload=json)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": '{"title":"ok"}'}]}}
                    ]
                },
            )

    monkeypatch.setattr(gemini_module.httpx, "AsyncClient", FakeAsyncClient)
    client = GeminiResponsesClient(api_key="gemini-test-key")

    response = await client.post(
        "ignored-openai-compatible-url",
        timeout=12,
        json={
            "model": "gemini-3.5-flash-lite",
            "max_output_tokens": 8000,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": "safe"}]},
                {"role": "user", "content": [{"type": "text", "text": "story"}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                }
            },
        },
    )

    assert response.status_code == 200
    assert captured["url"].endswith(
        "/v1beta/models/gemini-3.5-flash-lite:generateContent"
    )
    assert captured["headers"] == {"x-goog-api-key": "gemini-test-key"}
    generation_config = captured["payload"]["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["maxOutputTokens"] == 8000
    assert generation_config["responseJsonSchema"]["required"] == ["title"]
    assert not {"temperature", "topP", "topK"} & generation_config.keys()
