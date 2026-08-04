from __future__ import annotations

import base64
import binascii
import json
import struct
import zlib

import httpx
import pytest

from iread_ai.adapters.generation.gms_gemini_image import (
    GMSGeminiImageGenerator,
)
from iread_ai.ports.story_image_generator import (
    StoryImageProviderError,
    StoryImageReference,
)


def _png(label: str) -> bytes:
    red = sum(label.encode()) % 256
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes((0, red, 80, 160)))
    return (
        signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _reference(character_id: str, label: str) -> StoryImageReference:
    return StoryImageReference(
        character_id=character_id,
        name=character_id,
        description=f"{character_id} identity",
        content=_png(label),
        mime_type="image/png",
    )


@pytest.mark.asyncio
async def test_gemini_sends_prompt_ordered_server_references_and_21_by_9() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"x-goog-request-id": "gms-image-request-1"},
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(_png("RESULT")).decode(),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        generator = GMSGeminiImageGenerator(
            gms_key="secret",
            client=client,
        )
        result = await generator.generate(
            prompt="one continuous full-bleed story scene",
            references=[
                _reference("hare", "HARE"),
                _reference("tortoise", "TORTOISE"),
            ],
            aspect_ratio="21:9",
        )

    assert result.content == _png("RESULT")
    assert result.mime_type == "image/png"
    assert result.provider_request_id == "gms-image-request-1"
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["x-goog-api-key"] == "secret"
    payload = json.loads(request.content)
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"text": "one continuous full-bleed story scene"}
    assert [base64.b64decode(part["inline_data"]["data"]) for part in parts[1:]] == [
        _png("HARE"),
        _png("TORTOISE"),
    ]
    assert payload["generationConfig"]["responseFormat"]["image"] == {
        "aspectRatio": "ASPECT_RATIO_TWENTY_ONE_BY_NINE"
    }


@pytest.mark.asyncio
async def test_gemini_timeout_is_sanitized_and_retryable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        generator = GMSGeminiImageGenerator(
            gms_key="must-not-leak",
            client=client,
        )
        with pytest.raises(StoryImageProviderError) as raised:
            await generator.generate(
                prompt="safe prompt",
                references=[],
                aspect_ratio="21:9",
            )

    assert raised.value.code == "TIMEOUT"
    assert raised.value.retryable is True
    assert "must-not-leak" not in str(raised.value)


@pytest.mark.asyncio
async def test_gemini_rejects_non_image_response_data() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": base64.b64encode(b"not-an-image").decode(),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        generator = GMSGeminiImageGenerator(
            gms_key="secret",
            client=client,
        )
        with pytest.raises(
            StoryImageProviderError,
            match="unsupported image",
        ):
            await generator.generate(
                prompt="safe prompt",
                references=[],
                aspect_ratio="21:9",
            )
