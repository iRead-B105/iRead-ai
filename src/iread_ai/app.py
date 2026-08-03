from __future__ import annotations

import hmac
from html import escape
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .config import Settings
from .generation_models import (
    ContinueStoryRequest,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateStoryRequest,
    GenerateStoryResponse,
    GenerateTrainingRequest,
    GenerateTrainingResponse,
    TrainingCandidateRequest,
    TrainingCandidateResponse,
)
from .generation_service import generate_training
from .idempotency import (
    IdempotencyConflict,
    IdempotencyInProgress,
    MemoryIdempotencyStore,
)
from .mock_generators import (
    continue_story,
    generate_legacy_training,
    generate_story,
)
from .models import (
    PronunciationAnalysisResponse,
    TrainingEvaluateRequest,
    TrainingEvaluateResponse,
)
from .pronunciation import AzurePronunciationProvider, PronunciationProviderError
from .providers import GMSTextProvider


settings = Settings.from_env()
provider = AzurePronunciationProvider(settings)
idempotency_store = MemoryIdempotencyStore(
    ttl_seconds=settings.idempotency_ttl_seconds
)
text_provider = (
    GMSTextProvider(
        api_key=settings.gms_key,
        model=settings.gms_text_model,
        base_url=settings.gms_text_base_url,
        timeout_seconds=settings.gms_text_timeout_seconds,
        max_output_tokens=settings.gms_max_output_tokens,
    )
    if settings.generation_provider == "gms"
    else None
)
app = FastAPI(
    title="iRead AI",
    version="0.3.0",
    description=(
        "GMS 훈련 생성, 안전 대체 콘텐츠와 Azure 발음 평가를 제공합니다."
    ),
)


def _validate_idempotency(
    request_id: str,
    header_value: str | None,
) -> None:
    if header_value is not None and header_value != request_id:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key와 requestId가 일치해야 합니다.",
        )


def _authorize(
    x_api_key: str | None,
    idempotency_key: str | None,
) -> str:
    _require_api_key(x_api_key)
    if idempotency_key is None or not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="Idempotency-Key is required")
    key = idempotency_key.strip()
    if len(key) > 256:
        raise HTTPException(status_code=400, detail="Idempotency-Key is too long")
    return key


def _execute_idempotent(
    *,
    scope: str,
    key: str,
    payload: Any,
    action: Callable[[], Any],
) -> tuple[Any, bool]:
    try:
        return idempotency_store.execute(
            scope=scope,
            key=key,
            payload=payload,
            action=action,
        )
    except IdempotencyConflict as exception:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was reused with different request content",
        ) from exception
    except IdempotencyInProgress as exception:
        raise HTTPException(
            status_code=409,
            detail="request with this Idempotency-Key is still in progress",
        ) from exception


def _evaluate_training(
    request: TrainingEvaluateRequest,
) -> TrainingEvaluateResponse:
    questions = request.result.get("questions")
    if not isinstance(questions, list) or not questions:
        accuracy = 0.0
    else:
        correct = sum(
            1
            for question in questions
            if isinstance(question, dict) and question.get("isCorrect") is True
        )
        accuracy = round(correct * 100 / len(questions), 2)
    return TrainingEvaluateResponse(
        requestId=request.requestId,
        schemaVersion=request.schemaVersion,
        accuracy=accuracy,
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "UP", "service": "iread-ai"}


@app.post(
    "/api/v1/trainings/candidates",
    response_model=TrainingCandidateResponse,
    tags=["generation"],
    summary="34개 훈련 타입별 문항 후보 생성",
)
def training_candidates(
    request: TrainingCandidateRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> TrainingCandidateResponse:
    key = _authorize(x_api_key, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-candidates",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: generate_training(request, text_provider),
    )
    response.headers["X-AI-Provider"] = result.provider
    if result.fallback:
        response.headers["X-AI-Fallback"] = "safe-mock"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result.value


@app.post(
    "/api/v1/trainings/generate",
    response_model=GenerateTrainingResponse,
    tags=["generation"],
    summary="레거시 훈련 데이터 envelope 생성",
)
def training_generate(
    request: GenerateTrainingRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> GenerateTrainingResponse:
    key = _authorize(x_api_key, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-generate",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: generate_legacy_training(request),
    )
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result


@app.post(
    "/api/v1/trainings/evaluate",
    response_model=TrainingEvaluateResponse,
    tags=["generation"],
    summary="훈련 결과 정확도 평가",
)
def training_evaluate(
    request: TrainingEvaluateRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TrainingEvaluateResponse:
    key = _authorize(x_api_key, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-evaluate",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: _evaluate_training(request),
    )
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result


@app.post(
    "/api/v1/story/generate",
    response_model=GenerateStoryResponse,
    tags=["generation"],
    summary="이야기 최초 대사 1~5 생성",
)
def story_generate(
    request: GenerateStoryRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> GenerateStoryResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    return generate_story(request)


@app.post(
    "/api/v1/story/continue",
    response_model=GenerateStoryResponse,
    tags=["generation"],
    summary="분기 선택을 반영한 이야기 대사 6~10 생성",
)
def story_continue(
    request: ContinueStoryRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> GenerateStoryResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    return continue_story(request)


@app.post(
    "/api/v1/images/generate",
    response_model=GenerateImageResponse,
    tags=["generation"],
    summary="훈련 장면 또는 완료 이야기 주인공 이미지 생성",
)
def image_generate(
    request: GenerateImageRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> GenerateImageResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    kind = (
        "character"
        if request.prompt.strip().startswith("[STORY_CHARACTER]")
        else "scene"
    )
    query = urlencode({"label": request.prompt, "kind": kind})
    return GenerateImageResponse(
        requestId=request.requestId,
        imageUrl=f"/api/v1/images/mock/generated.svg?{query}",
        provider="IREAD_MOCK_AI_SVG_V1",
    )


@app.get(
    "/api/v1/images/mock/generated.svg",
    tags=["generation"],
    response_class=Response,
    summary="결정적 mock 이미지 조회",
)
def generated_image(
    label: str = Query(default="iRead 생성 이미지", max_length=300),
    kind: str = Query(default="scene", pattern="^(scene|character)$"),
) -> Response:
    display = label.replace("[STORY_CHARACTER]", "").strip()
    if len(display) > 42:
        display = display[:42] + "…"
    title = "이야기 친구" if kind == "character" else "훈련 장면"
    accent = "#7867c7" if kind == "character" else "#4c9f70"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
      <rect width="960" height="540" rx="32" fill="#f7f3ff"/>
      <circle cx="480" cy="220" r="118" fill="{accent}" opacity=".18"/>
      <circle cx="480" cy="210" r="76" fill="{accent}"/>
      <circle cx="452" cy="195" r="8" fill="#fff"/>
      <circle cx="508" cy="195" r="8" fill="#fff"/>
      <path d="M450 235 Q480 260 510 235" fill="none" stroke="#fff" stroke-width="9" stroke-linecap="round"/>
      <rect x="130" y="355" width="700" height="112" rx="30" fill="#fff"/>
      <text x="480" y="397" text-anchor="middle" font-family="sans-serif" font-size="26" font-weight="700" fill="{accent}">{escape(title)}</text>
      <text x="480" y="438" text-anchor="middle" font-family="sans-serif" font-size="21" fill="#4f4965">{escape(display)}</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


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
