from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from iread_ai.ports.story_image_generator import (
    GeneratedStoryImage,
    StoryImageProviderError,
    StoryImageReference,
)

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:/-]{1,256}$")
_ASPECT_RATIOS = {
    "1:1": "ASPECT_RATIO_ONE_BY_ONE",
    "21:9": "ASPECT_RATIO_TWENTY_ONE_BY_NINE",
}
_IMAGE_SIGNATURES: tuple[tuple[str, bytes], ...] = (
    ("image/png", b"\x89PNG\r\n\x1a\n"),
    ("image/jpeg", b"\xff\xd8\xff"),
)


class GMSGeminiImageGenerator:
    def __init__(
        self,
        *,
        gms_key: str,
        model: str = "gemini-2.5-flash-image",
        base_url: str = "https://gms.ssafy.io/gmsapi",
        timeout_seconds: float = 120.0,
        max_image_bytes: int = 12 * 1024 * 1024,
        max_response_bytes: int = 20 * 1024 * 1024,
        max_request_bytes: int = 20 * 1024 * 1024,
        max_prompt_bytes: int = 64 * 1024,
        client: httpx.AsyncClient | None = None,
        direct: bool = False,
    ) -> None:
        if not gms_key.strip():
            raise ValueError("gms_key must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        parsed_url = httpx.URL(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.host:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name, value in (
            ("max_image_bytes", max_image_bytes),
            ("max_response_bytes", max_response_bytes),
            ("max_request_bytes", max_request_bytes),
            ("max_prompt_bytes", max_prompt_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self._gms_key = gms_key
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_image_bytes = max_image_bytes
        self._max_response_bytes = max_response_bytes
        self._max_request_bytes = max_request_bytes
        self._max_prompt_bytes = max_prompt_bytes
        self._client = client
        root = base_url.rstrip("/")
        self._url = (
            f"{root}/v1beta/models/{model}:generateContent"
            if direct
            else f"{root}/generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

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
        prompt_bytes = prompt.encode("utf-8")
        if not prompt_bytes or len(prompt_bytes) > self._max_prompt_bytes:
            raise StoryImageProviderError(
                "INVALID_PROMPT",
                "Image prompt was empty or exceeded the configured size limit.",
                retryable=False,
            )
        try:
            aspect_ratio_enum = _ASPECT_RATIOS[aspect_ratio]
        except KeyError as exc:
            raise ValueError(
                f"aspect_ratio must be one of: {', '.join(sorted(_ASPECT_RATIOS))}"
            ) from exc

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for reference in references:
            _validate_reference(reference, self._max_image_bytes)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": reference.mime_type,
                        "data": base64.b64encode(reference.content).decode("ascii"),
                    }
                }
            )
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "responseFormat": {
                    "image": {"aspectRatio": aspect_ratio_enum},
                },
            },
        }
        request_size = len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if request_size >= self._max_request_bytes:
            raise StoryImageProviderError(
                "REQUEST_TOO_LARGE",
                "Gemini inline request exceeded the configured size limit.",
                retryable=False,
            )

        response = await self._post(payload)
        request_id = _request_id(response)
        if response.status_code >= 400:
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            raise StoryImageProviderError(
                "PROVIDER_HTTP_ERROR",
                f"Gemini image request failed (HTTP {response.status_code}).",
                retryable=retryable,
            )
        if len(response.content) > self._max_response_bytes:
            raise StoryImageProviderError(
                "RESPONSE_TOO_LARGE",
                "Gemini response exceeded the configured size limit.",
                retryable=False,
            )
        try:
            document = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise StoryImageProviderError(
                "INVALID_RESPONSE",
                "Gemini returned malformed JSON.",
                retryable=False,
            ) from exc
        if not isinstance(document, Mapping):
            raise StoryImageProviderError(
                "INVALID_RESPONSE",
                "Gemini returned an unexpected JSON document.",
                retryable=False,
            )
        inline_data = _find_inline_image(document)
        if inline_data is None:
            block_reason = _safe_block_reason(document)
            code = "SAFETY_BLOCKED" if block_reason else "NO_IMAGE"
            detail = f" ({block_reason})" if block_reason else ""
            raise StoryImageProviderError(
                code,
                f"Gemini completed without an image{detail}.",
                retryable=False,
            )
        image = _decode_image(
            inline_data.get("data"),
            inline_data.get("mime_type", inline_data.get("mimeType")),
            max_image_bytes=self._max_image_bytes,
        )
        return GeneratedStoryImage(
            content=image[0],
            mime_type=image[1],
            provider_request_id=request_id,
        )

    async def _post(self, payload: Mapping[str, Any]) -> httpx.Response:
        headers = {
            "x-goog-api-key": self._gms_key,
            "Content-Type": "application/json",
        }
        try:
            if self._client is not None:
                return await self._client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                return await client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise StoryImageProviderError(
                "TIMEOUT",
                "Gemini image request timed out.",
                retryable=True,
            ) from exc
        except httpx.RequestError as exc:
            raise StoryImageProviderError(
                "UNAVAILABLE",
                "Gemini image endpoint was unavailable.",
                retryable=True,
            ) from exc


def _validate_reference(
    reference: StoryImageReference,
    max_image_bytes: int,
) -> None:
    if not reference.content or len(reference.content) > max_image_bytes:
        raise StoryImageProviderError(
            "INVALID_REFERENCE",
            "A character reference was empty or exceeded the size limit.",
            retryable=False,
        )
    detected = _detect_mime(reference.content)
    if detected is None or detected != reference.mime_type:
        raise StoryImageProviderError(
            "INVALID_REFERENCE",
            "A character reference had an unsupported or mismatched image type.",
            retryable=False,
        )


def _find_inline_image(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = document.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(
        candidates,
        (str, bytes, bytearray),
    ):
        return None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts")
        if not isinstance(parts, Sequence) or isinstance(
            parts,
            (str, bytes, bytearray),
        ):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            inline_data = part.get("inline_data", part.get("inlineData"))
            if isinstance(inline_data, Mapping):
                return inline_data
    return None


def _safe_block_reason(document: Mapping[str, Any]) -> str | None:
    feedback = document.get("promptFeedback", document.get("prompt_feedback"))
    if not isinstance(feedback, Mapping):
        return None
    reason = feedback.get("blockReason", feedback.get("block_reason"))
    allowed = {
        "SAFETY",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "IMAGE_SAFETY",
    }
    return reason if isinstance(reason, str) and reason in allowed else None


def _decode_image(
    encoded: object,
    declared_mime: object,
    *,
    max_image_bytes: int,
) -> tuple[bytes, str]:
    if not isinstance(encoded, str) or not encoded:
        raise StoryImageProviderError(
            "INVALID_IMAGE_DATA",
            "Gemini response did not contain base64 image data.",
            retryable=False,
        )
    maximum_encoded_length = ((max_image_bytes + 2) // 3) * 4 + 4
    if len(encoded) > maximum_encoded_length:
        raise StoryImageProviderError(
            "IMAGE_TOO_LARGE",
            "Gemini image exceeded the configured size limit.",
            retryable=False,
        )
    try:
        content = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise StoryImageProviderError(
            "INVALID_IMAGE_DATA",
            "Gemini returned invalid base64 image data.",
            retryable=False,
        ) from exc
    if not content or len(content) > max_image_bytes:
        raise StoryImageProviderError(
            "IMAGE_TOO_LARGE",
            "Gemini image was empty or exceeded the configured size limit.",
            retryable=False,
        )
    detected_mime = _detect_mime(content)
    if detected_mime is None:
        raise StoryImageProviderError(
            "UNSUPPORTED_IMAGE",
            "Gemini returned an unsupported image type.",
            retryable=False,
        )
    normalized_declared = declared_mime.strip().lower() if isinstance(declared_mime, str) else None
    if normalized_declared == "image/jpg":
        normalized_declared = "image/jpeg"
    if normalized_declared is not None and normalized_declared != detected_mime:
        raise StoryImageProviderError(
            "IMAGE_MIME_MISMATCH",
            "Gemini image MIME type did not match its content.",
            retryable=False,
        )
    return content, detected_mime


def _detect_mime(content: bytes) -> str | None:
    for mime_type, signature in _IMAGE_SIGNATURES:
        if content.startswith(signature):
            return mime_type
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _request_id(response: httpx.Response) -> str | None:
    for header in ("x-request-id", "x-goog-request-id"):
        candidate = response.headers.get(header)
        if candidate is not None and _SAFE_REQUEST_ID.fullmatch(candidate):
            return candidate
    return None


__all__ = ["GMSGeminiImageGenerator"]
