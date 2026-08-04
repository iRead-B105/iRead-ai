from __future__ import annotations

import base64
from collections.abc import Sequence

import httpx

from iread_ai.ports.story_image_generator import (
    GeneratedStoryImage,
    StoryImageProviderError,
    StoryImageReference,
)


class OpenAIImageGenerator:
    def __init__(
        self, *, api_key: str, model: str, base_url: str, timeout_seconds: float = 180.0
    ) -> None:
        if not api_key.strip() or not model.strip():
            raise ValueError("OpenAI image credentials and model must not be empty")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        *,
        prompt: str,
        references: Sequence[StoryImageReference],
        aspect_ratio: str,
    ) -> GeneratedStoryImage:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                if references:
                    files = [
                        (
                            "image[]",
                            (f"reference-{index}.png", reference.content, reference.mime_type),
                        )
                        for index, reference in enumerate(references)
                    ]
                    response = await client.post(
                        f"{self._base_url}/images/edits",
                        headers=headers,
                        data={
                            "model": self._model,
                            "prompt": prompt,
                            "size": "1536x1024",
                            "response_format": "b64_json",
                        },
                        files=files,
                    )
                else:
                    response = await client.post(
                        f"{self._base_url}/images/generations",
                        headers={**headers, "Content-Type": "application/json"},
                        json={
                            "model": self._model,
                            "prompt": prompt,
                            "size": "1536x1024",
                            "response_format": "b64_json",
                        },
                    )
        except (httpx.RequestError, TimeoutError) as exc:
            raise StoryImageProviderError(
                "UNAVAILABLE", "OpenAI image endpoint was unavailable.", retryable=True
            ) from exc
        if response.status_code >= 400:
            raise StoryImageProviderError(
                "PROVIDER_HTTP_ERROR",
                f"OpenAI image request failed (HTTP {response.status_code}).",
                retryable=response.status_code in {408, 409, 429} or response.status_code >= 500,
            )
        try:
            encoded = response.json()["data"][0]["b64_json"]
            content = base64.b64decode(encoded, validate=True)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StoryImageProviderError(
                "INVALID_RESPONSE", "OpenAI returned invalid image data.", retryable=False
            ) from exc
        if not content:
            raise StoryImageProviderError(
                "NO_IMAGE", "OpenAI completed without an image.", retryable=False
            )
        return GeneratedStoryImage(
            content=content,
            mime_type="image/png",
            provider_request_id=response.headers.get("x-request-id"),
        )


__all__ = ["OpenAIImageGenerator"]
