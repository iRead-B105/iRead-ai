from __future__ import annotations

import base64
import hashlib
import hmac
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from .adapters.generation.gms_gemini_image import GMSGeminiImageGenerator
from .config import Settings
from .curriculum_models import CurriculumRecommendRequest, CurriculumRecommendResponse
from .curriculum_recommender import recommend_curriculum
from .generation_models import (
    EvaluateTrainingRequest,
    EvaluateTrainingResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateTrainingRequest,
    GenerateTrainingResponse,
    SpeechSynthesisRequest,
    SpeechTranscriptionResponse,
    TrainingActivityRequest,
    TrainingActivityResponse,
    TrainingCandidateRequest,
    TrainingCandidateResponse,
    TrainingSetRequest,
    TrainingSetResponse,
)
from .generation_service import enrich_training_request_with_lexicon, generate_training
from .idempotency import (
    IdempotencyConflict,
    IdempotencyInProgress,
)
from .idempotency import (
    MemoryIdempotencyStore as TrainingIdempotencyStore,
)
from .main import create_app
from .mock_generators import generate_legacy_training
from .models import PronunciationAnalysisResponse
from .ports.story_image_generator import StoryImageProviderError
from .pronunciation import (
    AzurePronunciationProvider,
    DeterministicPronunciationProvider,
    GMSPronunciationProvider,
    PronunciationProviderError,
)
from .providers import GMSTextProvider
from .speech import (
    AzureSpeechProvider,
    DeterministicSpeechProvider,
    SpeechProviderError,
)
from .training_evaluation import TrainingEvaluationError, evaluate_training
from .training_set_service import generate_training_activity, generate_training_set

settings = Settings.from_env()
if settings.pronunciation_provider == "azure":
    provider = AzurePronunciationProvider(settings)
elif settings.pronunciation_provider == "gms":
    provider = GMSPronunciationProvider(settings)
else:
    provider = DeterministicPronunciationProvider()
speech_provider = (
    AzureSpeechProvider(settings)
    if settings.speech_provider == "azure"
    else DeterministicSpeechProvider()
)
idempotency_store = TrainingIdempotencyStore(
    ttl_seconds=settings.idempotency_ttl_seconds
)
text_provider = (
    GMSTextProvider(
        api_key=settings.gms_key.get_secret_value(),
        model=settings.gms_text_model,
        base_url=settings.gms_text_base_url,
        timeout_seconds=settings.gms_text_timeout_seconds,
        max_output_tokens=settings.gms_max_output_tokens,
    )
    if settings.generation_provider == "gms" and settings.gms_key is not None
    else None
)
legacy_image_generator = (
    GMSGeminiImageGenerator(
        gms_key=settings.gms_key.get_secret_value(),
        model=settings.gms_gemini_image_model,
        base_url=settings.gms_base_url,
        timeout_seconds=settings.gms_image_timeout_seconds,
        max_image_bytes=settings.gms_image_max_bytes,
        max_response_bytes=settings.gms_image_max_response_bytes,
        max_request_bytes=settings.gms_image_max_request_bytes,
    )
    if settings.story_image_provider == "gemini" and settings.gms_key is not None
    else None
)
_generated_images: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_MAX_GENERATED_IMAGES = 128
_LEGACY_IMAGE_POLICY = (
    Path(__file__).resolve().parent / "prompts" / "story_image_legacy.md"
).read_text(encoding="utf-8")

# Personalized story, chapter, and Gemini image routes are assembled by the
# canonical app factory. The legacy training and speech contracts below remain
# available to Backend while it migrates to the richer contracts.
app = create_app(settings=settings)


def _require_api_key(value: str | None) -> None:
    configured = settings.internal_api_key
    expected = (
        configured.get_secret_value()
        if hasattr(configured, "get_secret_value")
        else str(configured)
    )
    if value is None or not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


def _validate_idempotency(request_id: str, header_value: str | None) -> str:
    if header_value is None or not header_value.strip():
        raise HTTPException(status_code=400, detail="Idempotency-Key is required")
    if header_value != request_id:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key와 requestId가 일치해야 합니다.",
        )
    return header_value


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


@app.post(
    "/api/v1/trainings/candidates",
    response_model=TrainingCandidateResponse,
    tags=["generation"],
    summary="34개 훈련 타입별 문항 후보 생성",
)
def training_candidates(
    request: TrainingCandidateRequest,
    response: Response,
    http_request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TrainingCandidateResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-candidates",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: generate_training(
            enrich_training_request_with_lexicon(
                request,
                http_request.app.state.lexicon_service,
            ),
            text_provider,
        ),
    )
    response.headers["X-AI-Provider"] = result.provider
    if result.fallback:
        response.headers["X-AI-Fallback"] = "curated-fallback"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result.value


@app.post(
    "/api/v1/training-sets/generate",
    response_model=TrainingSetResponse,
    tags=["generation"],
    summary="한 학습 목표를 서로 다른 5개 훈련 활동으로 구성",
)
def training_set_generate(
    request: TrainingSetRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TrainingSetResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-set-generate",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: generate_training_set(
            request,
            text_provider,
            lexicon_service=app.state.lexicon_service,
            analyzer=app.state.story_chapter_analyzer,
        ),
    )
    response.headers["X-AI-Provider"] = "mixed:" + ",".join(
        sorted(set(result.providers))
    )
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result.response


@app.post(
    "/api/v1/curricula/recommend",
    response_model=CurriculumRecommendResponse,
    tags=["recommendation"],
    summary="학생의 읽기 단계와 취약 특징에 맞는 다음 훈련 5개 추천",
)
def curriculum_recommend(
    request: CurriculumRecommendRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> CurriculumRecommendResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="curriculum-recommend",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: recommend_curriculum(request, text_provider),
    )
    response.headers["X-AI-Provider"] = result.recommendationProvider
    if result.recommendationProvider == "deterministic-fallback":
        response.headers["X-AI-Fallback"] = "stage-gated-deterministic"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result


@app.post(
    "/api/v1/training-activities/generate",
    response_model=TrainingActivityResponse,
    tags=["generation"],
    summary="맞춤 훈련 세트의 활동 하나를 재생성",
)
def training_activity_generate(
    request: TrainingActivityRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> TrainingActivityResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
    result, replayed = _execute_idempotent(
        scope="training-activity-generate",
        key=key,
        payload=request.model_dump(mode="json"),
        action=lambda: generate_training_activity(
            request,
            text_provider,
            lexicon_service=app.state.lexicon_service,
            analyzer=app.state.story_chapter_analyzer,
        ),
    )
    response.headers["X-AI-Provider"] = result.activity.provider
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result


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
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> GenerateTrainingResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
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
    response_model=EvaluateTrainingResponse,
    tags=["generation"],
    summary="훈련 결과 정확도 평가(결정적 규칙)",
)
def training_evaluate(
    request: EvaluateTrainingRequest,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> EvaluateTrainingResponse:
    _require_api_key(x_api_key)
    key = _validate_idempotency(request.requestId, idempotency_key)
    try:
        result, replayed = _execute_idempotent(
            scope="training-evaluate",
            key=key,
            payload=request.model_dump(mode="json"),
            action=lambda: evaluate_training(request),
        )
    except TrainingEvaluationError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception
    response.headers["X-AI-Provider"] = "hybrid-evaluator"
    if replayed:
        response.headers["Idempotent-Replayed"] = "true"
    return result


@app.post(
    "/api/v1/images/generate",
    response_model=GenerateImageResponse,
    tags=["generation"],
    summary="레거시 이미지 생성 호환 API",
)
async def image_generate(
    request: GenerateImageRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> GenerateImageResponse:
    _require_api_key(x_api_key)
    _validate_idempotency(request.requestId, idempotency_key)
    kind = "character" if request.prompt.strip().startswith("[STORY_CHARACTER]") else "scene"
    digest = hashlib.sha256(
        f"{request.requestId}\0{_LEGACY_IMAGE_POLICY}\0{request.prompt}".encode()
    ).hexdigest()[:32]
    if legacy_image_generator is not None and digest not in _generated_images:
        prompt = (
            f"{_LEGACY_IMAGE_POLICY}\n\n"
            f"[UNTRUSTED STORY SCENE DATA]\n{request.prompt.strip()}"
        )
        try:
            generated = await legacy_image_generator.generate(
                prompt=prompt,
                references=(),
                aspect_ratio="1:1" if kind == "character" else "21:9",
            )
        except StoryImageProviderError as exception:
            raise HTTPException(
                status_code=503 if exception.retryable else 502,
                detail=exception.safe_message,
            ) from exception
        _generated_images[digest] = (generated.content, generated.mime_type)
        _generated_images.move_to_end(digest)
        while len(_generated_images) > _MAX_GENERATED_IMAGES:
            _generated_images.popitem(last=False)
    query = urlencode({"label": digest, "kind": kind})
    return GenerateImageResponse(
        requestId=request.requestId,
        imageUrl=f"/api/v1/images/generated.png?{query}",
        provider=(
            f"GMS_GEMINI_{legacy_image_generator.model}"
            if legacy_image_generator is not None
            else "IREAD_MOCK_AI_PNG_V1"
        ),
    )


_MOCK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@app.get(
    "/api/v1/images/generated.png",
    tags=["generation"],
    response_class=Response,
    summary="생성 이미지 조회",
)
@app.get(
    "/api/v1/images/mock/generated.png",
    tags=["generation"],
    response_class=Response,
    summary="결정적 mock PNG 조회",
)
def generated_image(
    label: str = Query(default="iread", max_length=64),
    kind: str = Query(default="scene", pattern="^(scene|character)$"),
) -> Response:
    del kind
    generated = _generated_images.get(label)
    if generated is None:
        return Response(content=_MOCK_PNG, media_type="image/png")
    _generated_images.move_to_end(label)
    content, mime_type = generated
    return Response(content=content, media_type=mime_type)


@app.post(
    "/api/v1/speech/pronunciation/analyze",
    response_model=PronunciationAnalysisResponse,
    tags=["speech"],
)
async def analyze_pronunciation(
    requestId: str = Form(min_length=1),
    expectedText: str = Form(min_length=1),
    audioFile: UploadFile = File(),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> PronunciationAnalysisResponse:
    _require_api_key(x_api_key)
    _validate_idempotency(requestId, idempotency_key)
    audio = await audioFile.read(settings.max_audio_bytes + 1)
    filename = audioFile.filename
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
            original_filename=filename,
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
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SpeechTranscriptionResponse:
    _require_api_key(x_api_key)
    _validate_idempotency(requestId, idempotency_key)
    audio = await audioFile.read(settings.max_audio_bytes + 1)
    filename = audioFile.filename
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
            original_filename=filename,
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
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Response:
    _require_api_key(x_api_key)
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
