from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GeminiResponsesClient:
    """Translate the small OpenAI Responses subset used by iRead to Gemini generateContent."""

    def __init__(
        self, *, api_key: str, base_url: str = "https://generativelanguage.googleapis.com"
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def post(  # noqa: ASYNC109 - mirrors httpx.AsyncClient.post
        self,
        _url: str,
        *,
        json: Mapping[str, Any],
        timeout: float,  # noqa: ASYNC109
        **_: Any,
    ) -> httpx.Response:
        model = str(json["model"])
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for message in json.get("input", []):
            if not isinstance(message, Mapping):
                continue
            parts = [
                {"text": str(part.get("text", ""))}
                for part in message.get("content", [])
                if isinstance(part, Mapping) and part.get("text")
            ]
            if message.get("role") == "system":
                system_parts.extend(parts)
            elif parts:
                contents.append({"role": "user", "parts": parts})
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": json.get("max_output_tokens", 2400),
        }
        text_config = json.get("text")
        if isinstance(text_config, Mapping):
            output_format = text_config.get("format")
            if isinstance(output_format, Mapping) and isinstance(
                output_format.get("schema"), Mapping
            ):
                generation_config["responseJsonSchema"] = output_format["schema"]
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        endpoint = f"{self._base_url}/v1beta/models/{model}:generateContent"
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                headers={
                    "x-goog-api-key": self._api_key,
                    "Accept-Encoding": "identity",
                },
                json=payload,
            )
        if response.status_code >= 400:
            error_code: object = None
            error_status: object = None
            try:
                error_document = response.json()
                error = error_document.get("error", {})
                if isinstance(error, Mapping):
                    error_code = error.get("code")
                    error_status = error.get("status")
            except (ValueError, TypeError):
                pass
            logger.warning(
                "Gemini text request failed model=%s http_status=%s "
                "error_code=%s error_status=%s",
                model,
                response.status_code,
                error_code,
                error_status,
            )
            return response
        document = response.json()
        candidates = document.get("candidates", []) if isinstance(document, Mapping) else []
        text = ""
        if isinstance(candidates, Sequence) and candidates and isinstance(candidates[0], Mapping):
            content = candidates[0].get("content", {})
            parts = content.get("parts", []) if isinstance(content, Mapping) else []
            text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, Mapping))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            },
            headers=response.headers,
        )


__all__ = ["GeminiResponsesClient"]
