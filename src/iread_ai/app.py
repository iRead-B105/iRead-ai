from __future__ import annotations

import hmac
from html import escape
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .config import Settings
from .generation_models import (
    ContinueStoryRequest,
    EvaluateTrainingRequest,
    EvaluateTrainingResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateStoryRequest,
    GenerateStoryResponse,
    GenerateTrainingRequest,
    GenerateTrainingResponse,
    SpeechSynthesisRequest,
    SpeechTranscriptionResponse,
    TrainingCandidateRequest,
    TrainingCandidateResponse,
)
from .mock_generators import (
    continue_story,
    evaluate_training,
    generate_legacy_training,
    generate_story,
    generate_training_candidates,
)
from .models import PronunciationAnalysisResponse
from .pronunciation import (
    AzurePronunciationProvider,
    DeterministicPronunciationProvider,
    PronunciationProviderError,
)
from .speech import (
    AzureSpeechProvider,
    DeterministicSpeechProvider,
    SpeechProviderError,
)


settings = Settings.from_env()
provider = (
    AzurePronunciationProvider(settings)
    if settings.pronunciation_provider == "azure"
    else DeterministicPronunciationProvider()
)
speech_provider = (
    AzureSpeechProvider(settings)
    if settings.speech_provider == "azure"
    else DeterministicSpeechProvider()
)
app = FastAPI(
    title="iRead AI",
    version="0.2.0",
    description=(
        "Azure 발음 평가와 백엔드 연동용 결정적 생성 mock을 함께 제공합니다."
    ),
)


@app.middleware("http")
async def require_internal_api_key(request: Request, call_next):
    """Protect every mutable internal AI endpoint with the shared backend key."""
    if request.method != "GET" and request.url.path.startswith("/api/v1/"):
        expected = settings.internal_api_key
        supplied = request.headers.get("X-API-Key")
        if not expected or supplied is None or not hmac.compare_digest(supplied, expected):
            return JSONResponse(status_code=401, content={"detail": "invalid API key"})
    return await call_next(request)


def _validate_idempotency(
    request_id: str,
    header_value: str | None,
) -> None:
    if header_value is not None and header_value != request_id:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key와 requestId가 일치해야 합니다.",
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
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> TrainingCandidateResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    try:
        return generate_training_candidates(request)
    except ValueError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception


@app.post(
    "/api/v1/trainings/generate",
    response_model=GenerateTrainingResponse,
    tags=["generation"],
    summary="레거시 훈련 데이터 envelope 생성",
)
def training_generate(
    request: GenerateTrainingRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> GenerateTrainingResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    return generate_legacy_training(request)


@app.post(
    "/api/v1/trainings/evaluate",
    response_model=EvaluateTrainingResponse,
    tags=["generation"],
    summary="훈련 결과 정확도 평가(결정적 mock)",
)
def training_evaluate(
    request: EvaluateTrainingRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> EvaluateTrainingResponse:
    _validate_idempotency(request.requestId, idempotency_key)
    return evaluate_training(request)


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


@app.post(
    "/api/v1/speech/transcribe",
    response_model=SpeechTranscriptionResponse,
    tags=["speech"],
    summary="음성 인식(STT)",
)
async def transcribe_speech(
    requestId: str = Form(min_length=1),
    studentId: int = Form(ge=1),
    expectedText: str | None = Form(default=None),
    audioFile: UploadFile = File(),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> SpeechTranscriptionResponse:
    _validate_idempotency(requestId, idempotency_key)
    audio = await audioFile.read(settings.max_audio_bytes + 1)
    await audioFile.close()
    if not audio:
        raise HTTPException(status_code=400, detail="audioFile is empty")
    if len(audio) > settings.max_audio_bytes:
        raise HTTPException(status_code=413, detail="audioFile is too large")
    del studentId, expectedText
    try:
        return speech_provider.transcribe(
            request_id=requestId,
            audio=audio,
            original_filename=audioFile.filename,
        )
    except SpeechProviderError as exception:
        raise HTTPException(status_code=502, detail=str(exception)) from exception


@app.post(
    "/api/v1/speech/synthesize",
    tags=["speech"],
    response_class=Response,
    summary="음성 합성(TTS)",
)
def synthesize_speech(
    request: SpeechSynthesisRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> Response:
    _validate_idempotency(request.requestId, idempotency_key)
    try:
        result = speech_provider.synthesize(request)
    except SpeechProviderError as exception:
        raise HTTPException(status_code=502, detail=str(exception)) from exception
    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={
            "X-Request-Id": request.requestId,
            "X-Audio-Duration-Ms": str(result.duration_ms),
        },
    )


def _require_api_key(value: str | None) -> None:
    expected = settings.internal_api_key
    if not expected or value is None or not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="invalid API key")
