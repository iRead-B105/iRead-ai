from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from iread_ai.adapters.generation.gemini_responses import GeminiResponsesClient
from iread_ai.adapters.generation.gms_gemini_image import (
    GMSGeminiImageGenerator,
)
from iread_ai.adapters.generation.gms_teacher_report import (
    GMSTeacherReportNarrator,
)
from iread_ai.adapters.generation.openai_image import OpenAIImageGenerator
from iread_ai.adapters.idempotency.memory import MemoryIdempotencyStore
from iread_ai.api.branch_input_review import router as branch_input_review_router
from iread_ai.api.comparison import router as comparison_router
from iread_ai.api.errors import install_error_handlers
from iread_ai.api.legacy_story import router as legacy_story_router
from iread_ai.api.lexicon import router as lexicon_router
from iread_ai.api.story_chapter import router as story_chapter_router
from iread_ai.api.story_image import router as story_image_router
from iread_ai.api.teacher_report import router as teacher_report_router
from iread_ai.application.branch_input_review import (
    BranchInputReviewer,
    DeterministicBranchInputReviewer,
    OpenAICompatibleBranchInputReviewer,
)
from iread_ai.application.legacy_story_service import LegacyStoryGenerationService
from iread_ai.application.personalized_chapter_service import (
    PersonalizedStoryChapterService,
)
from iread_ai.application.story_image_service import (
    KnownCharacterReferenceRepository,
    StoryImageApplicationService,
)
from iread_ai.application.teacher_report_summary_service import (
    TeacherReportSummaryService,
)
from iread_ai.config import Settings, get_settings
from iread_ai.lexicon.service import LexiconPaletteService
from iread_ai.personalization.analyzer import KoreanReadingAnalyzer
from iread_ai.personalization.chapter_comparison import (
    ChapterGenerationComparisonService,
)
from iread_ai.personalization.chapter_generator import (
    MockChapterCandidateGenerator,
    OpenAIChapterCandidateGenerator,
)
from iread_ai.personalization.generator import (
    MockPageCandidateGenerator,
    OpenAIPageCandidateGenerator,
)
from iread_ai.personalization.prompts import BASELINE_PROMPT_MODE
from iread_ai.personalization.visual_scene import (
    MockVisualScenePlanner,
    OpenAIVisualScenePlanner,
    VisualScenePlanner,
)
from iread_ai.ports.idempotency_store import IdempotencyStore
from iread_ai.providers import GMSTextProvider


def create_app(
    *,
    settings: Settings | None = None,
    idempotency_store: IdempotencyStore | None = None,
    chapter_generation_comparison_service: (ChapterGenerationComparisonService | None) = None,
    story_chapter_service: PersonalizedStoryChapterService | None = None,
    story_image_service: StoryImageApplicationService | None = None,
    teacher_report_service: TeacherReportSummaryService | None = None,
    lexicon_service: LexiconPaletteService | None = None,
    branch_input_reviewer: BranchInputReviewer | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    idempotency_store = idempotency_store or MemoryIdempotencyStore(
        ttl_seconds=settings.idempotency_ttl_seconds
    )
    story_image_service = story_image_service or _build_story_image_service(settings)
    teacher_report_service = teacher_report_service or _build_teacher_report_service(
        settings
    )
    lexicon_service = lexicon_service or LexiconPaletteService(
        settings.lexicon_database_path
    )
    branch_input_reviewer = branch_input_reviewer or _build_branch_input_reviewer(settings)

    legacy_story_chapter_service = story_chapter_service
    chapter_analyzer: KoreanReadingAnalyzer | None = None
    chapter_generator: MockChapterCandidateGenerator | OpenAIChapterCandidateGenerator | None = None
    if story_chapter_service is None:
        chapter_analyzer = KoreanReadingAnalyzer()
        chapter_repairer = _build_page_candidate_generator(settings)
        chapter_generator = _build_chapter_candidate_generator(settings)
        visual_scene_planner = _build_visual_scene_planner(settings)
        story_chapter_service = PersonalizedStoryChapterService(
            generator=chapter_generator,
            analyzer=chapter_analyzer,
            repairer=chapter_repairer,
            candidate_count=settings.chapter_candidate_count,
            quality_retry_count=settings.chapter_quality_retry_count,
            require_contract_pass=settings.chapter_require_contract_pass,
            provider_name=settings.story_provider,
            visual_scene_planner=visual_scene_planner,
        )
        legacy_story_chapter_service = PersonalizedStoryChapterService(
            generator=chapter_generator,
            analyzer=chapter_analyzer,
            repairer=chapter_repairer,
            candidate_count=settings.chapter_candidate_count,
            quality_retry_count=settings.chapter_quality_retry_count,
            require_contract_pass=settings.chapter_require_contract_pass,
            provider_name=settings.story_provider,
            visual_scene_planner=MockVisualScenePlanner(),
        )

    chapter_comparison_analyzer: KoreanReadingAnalyzer | None = None
    if chapter_generation_comparison_service is None and settings.app_env != "production":
        chapter_comparison_analyzer = chapter_analyzer or KoreanReadingAnalyzer()
        if chapter_generator is None:
            chapter_generator = _build_chapter_candidate_generator(settings)
        baseline_chapter_service = PersonalizedStoryChapterService(
            generator=chapter_generator,
            analyzer=chapter_comparison_analyzer,
            repairer=None,
            candidate_count=1,
            provider_name=settings.story_provider,
            prompt_mode=BASELINE_PROMPT_MODE,
        )
        chapter_generation_comparison_service = ChapterGenerationComparisonService(
            baseline_service=baseline_chapter_service,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.app_env != "test":
            analyzers = {
                id(analyzer): analyzer
                for analyzer in (
                    chapter_analyzer,
                    chapter_comparison_analyzer,
                )
                if analyzer is not None
            }
            for analyzer in analyzers.values():
                await asyncio.to_thread(analyzer.warmup)
        try:
            yield
        finally:
            lexicon_service.close()

    app = FastAPI(
        title="iRead AI",
        version="0.2.0",
        description="Backend 전용 iRead AI 내부 API",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.idempotency_store = idempotency_store
    app.state.story_chapter_service = story_chapter_service
    assert legacy_story_chapter_service is not None
    app.state.legacy_story_service = LegacyStoryGenerationService(
        chapter_service=legacy_story_chapter_service,
    )
    app.state.story_chapter_analyzer = chapter_analyzer
    app.state.story_image_service = story_image_service
    app.state.teacher_report_service = teacher_report_service
    app.state.lexicon_service = lexicon_service
    app.state.branch_input_reviewer = branch_input_reviewer
    app.state.chapter_generation_comparison_service = chapter_generation_comparison_service
    install_error_handlers(app)
    app.include_router(legacy_story_router)
    app.include_router(branch_input_review_router)
    app.include_router(story_chapter_router)
    app.include_router(story_image_router)
    app.include_router(teacher_report_router)
    app.include_router(lexicon_router)
    app.include_router(comparison_router)

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {
            "status": "UP",
            "service": "iread-ai",
            "storyProvider": settings.story_provider,
            "trainingGenerationProvider": settings.generation_provider,
            "storyImageProvider": settings.story_image_provider,
            "storyTextModel": settings.openai_model,
            "storyImageModel": settings.gms_gemini_image_model,
            "teacherReportProvider": (
                settings.generation_provider
                if settings.generation_provider in {"gms", "openai", "gemini"}
                else "deterministic"
            ),
            "lexiconStatus": lexicon_service.status().status,
        }

    return app


def _build_page_candidate_generator(
    settings: Settings,
) -> MockPageCandidateGenerator | OpenAIPageCandidateGenerator:
    if settings.story_provider == "mock":
        return MockPageCandidateGenerator()
    return OpenAIPageCandidateGenerator(
        api_key=settings.story_api_key,
        model=settings.openai_model,
        base_url=settings.story_api_base_url,
        timeout_seconds=settings.model_timeout_seconds,
        max_output_tokens=settings.openai_max_output_tokens,
        client=_story_http_client(settings),
    )


def _build_branch_input_reviewer(settings: Settings) -> BranchInputReviewer:
    if settings.story_provider == "mock":
        return DeterministicBranchInputReviewer()
    return OpenAICompatibleBranchInputReviewer(
        api_key=settings.story_api_key,
        model=settings.openai_model,
        base_url=settings.story_api_base_url,
        timeout_seconds=settings.branch_review_timeout_seconds,
        max_output_tokens=settings.branch_review_max_output_tokens,
        client=_story_http_client(settings),
    )


def _build_chapter_candidate_generator(
    settings: Settings,
) -> MockChapterCandidateGenerator | OpenAIChapterCandidateGenerator:
    if settings.story_provider == "mock":
        return MockChapterCandidateGenerator()
    return OpenAIChapterCandidateGenerator(
        api_key=settings.story_api_key,
        model=settings.openai_model,
        base_url=settings.story_api_base_url,
        timeout_seconds=settings.model_timeout_seconds,
        max_output_tokens=settings.openai_chapter_max_output_tokens,
        client=_story_http_client(settings),
    )


def _build_visual_scene_planner(
    settings: Settings,
) -> VisualScenePlanner:
    if settings.story_provider == "mock":
        return MockVisualScenePlanner()
    return OpenAIVisualScenePlanner(
        api_key=settings.story_api_key,
        model=settings.openai_model,
        base_url=settings.story_api_base_url,
        timeout_seconds=settings.visual_scene_timeout_seconds,
        max_output_tokens=settings.openai_visual_scene_max_output_tokens,
        client=_story_http_client(settings),
    )


def _story_http_client(settings: Settings):
    if settings.story_provider != "gemini":
        return None
    assert settings.gemini_api_key is not None
    return GeminiResponsesClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        base_url=settings.gemini_base_url,
    )


def _build_story_image_service(
    settings: Settings,
) -> StoryImageApplicationService:
    generator = None
    if settings.story_image_provider in {"gms", "gemini"}:
        image_key = (
            settings.gms_key if settings.story_image_provider == "gms" else settings.gemini_api_key
        )
        assert image_key is not None
        generator = GMSGeminiImageGenerator(
            gms_key=image_key.get_secret_value(),
            model=settings.gms_gemini_image_model,
            base_url=settings.gms_base_url
            if settings.story_image_provider == "gms"
            else settings.gemini_base_url,
            direct=settings.story_image_provider == "gemini",
            timeout_seconds=settings.gms_image_timeout_seconds,
            max_image_bytes=settings.gms_image_max_bytes,
            max_response_bytes=settings.gms_image_max_response_bytes,
            max_request_bytes=settings.gms_image_max_request_bytes,
        )
    elif settings.story_image_provider == "openai":
        assert settings.openai_api_key is not None
        generator = OpenAIImageGenerator(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.gms_gemini_image_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.gms_image_timeout_seconds,
        )
    return StoryImageApplicationService(
        generator=generator,
        references=KnownCharacterReferenceRepository(
            root=settings.character_reference_dir,
        ),
    )


def _build_teacher_report_service(
    settings: Settings,
) -> TeacherReportSummaryService:
    narrator = None
    if settings.generation_provider in {"gms", "openai", "gemini"}:
        narrator = GMSTeacherReportNarrator(
            GMSTextProvider(
                api_key=settings.text_api_key,
                model=settings.openai_model,
                base_url=settings.text_base_url,
                timeout_seconds=settings.text_timeout_seconds,
                max_output_tokens=settings.text_max_output_tokens,
                provider_name=settings.generation_provider,
            )
        )
    return TeacherReportSummaryService(narrator=narrator)
