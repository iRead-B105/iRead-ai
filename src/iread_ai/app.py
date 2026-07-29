from __future__ import annotations

import hmac

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from .config import Settings
from .models import PronunciationAnalysisResponse
from .pronunciation import AzurePronunciationProvider, PronunciationProviderError


settings = Settings.from_env()
provider = AzurePronunciationProvider(settings)
app = FastAPI(title="iRead AI", version="0.1.0")


@app.post(
    "/api/v1/speech/pronunciation/analyze",
    response_model=PronunciationAnalysisResponse,
)
async def analyze_pronunciation(
    requestId: str = Form(min_length=1),
    expectedText: str = Form(min_length=1),
    audioFile: UploadFile = File(),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> PronunciationAnalysisResponse:
    _require_api_key(x_api_key)
    if idempotency_key != requestId:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key must match requestId",
        )
    audio = await audioFile.read(settings.max_audio_bytes + 1)
    await audioFile.close()
    if not audio:
        raise HTTPException(status_code=400, detail="audioFile is empty")
    if len(audio) > settings.max_audio_bytes:
        raise HTTPException(status_code=413, detail="audioFile is too large")
    try:
        return provider.analyze(
            request_id=requestId,
            reference_text=expectedText,
            audio=audio,
            original_filename=audioFile.filename,
        )
    except PronunciationProviderError as exception:
        raise HTTPException(status_code=502, detail=str(exception)) from exception


def _require_api_key(value: str | None) -> None:
    expected = settings.internal_api_key
    if not expected or value is None or not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="invalid API key")
