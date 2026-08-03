"""GMS text and image adapters with bounded, contract-safe responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class GenerationProviderError(RuntimeError):
    """Safe provider error that never includes credentials or generated content."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class GMSTextProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float,
        max_output_tokens: int,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("GMS text credentials and model must not be empty")
        parsed = httpx.URL(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("GMS text base URL must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0 or max_output_tokens < 256:
            raise ValueError("GMS text limits are invalid")
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/responses"
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
            "max_output_tokens": self._max_output_tokens,
        }
        response = self._post(payload)
        if response.status_code >= 400:
            raise GenerationProviderError(
                f"GMS text request failed with HTTP {response.status_code}",
                retryable=_retryable_status(response.status_code),
            )
        try:
            document = response.json()
            if not isinstance(document, Mapping):
                raise TypeError("response root is not an object")
            if document.get("status") not in {None, "completed"}:
                raise ValueError("response did not complete")
            output = json.loads(_extract_output_text(document))
            if not isinstance(output, dict):
                raise TypeError("generated JSON root is not an object")
            return output
        except (json.JSONDecodeError, TypeError, ValueError) as exception:
            raise GenerationProviderError(
                "GMS text response did not match the requested JSON contract",
                retryable=False,
            ) from exception

    def _post(self, payload: Mapping[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self._client is not None:
                return self._client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            with httpx.Client(timeout=self._timeout_seconds) as client:
                return client.post(self._url, headers=headers, json=payload)
        except (httpx.TimeoutException, TimeoutError) as exception:
            raise GenerationProviderError(
                "GMS text request timed out", retryable=True
            ) from exception
        except httpx.RequestError as exception:
            raise GenerationProviderError(
                "GMS text endpoint is unavailable", retryable=True
            ) from exception


def _extract_output_text(document: Mapping[str, Any]) -> str:
    direct = document.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    chunks: list[str] = []
    output = document.get("output", [])
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise ValueError("response contained no output array")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") == "refusal":
                raise ValueError("model refused the request")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    result = "".join(chunks)
    if not result:
        raise ValueError("response contained no output text")
    return result


def _retryable_status(status: int) -> bool:
    return status in {408, 409, 429} or status >= 500
