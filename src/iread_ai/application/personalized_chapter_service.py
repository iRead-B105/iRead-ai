from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

from iread_ai.application.candidate_evaluation import (
    evaluate_page_candidate_resilient,
)
from iread_ai.contracts.story_chapter import (
    StoryChapterGenerateRequest,
    StoryChapterGenerateResponse,
)
from iread_ai.personalization.analyzer import (
    AnalysisStatus,
    CandidateAnalysis,
    KoreanReadingAnalyzer,
)
from iread_ai.personalization.chapter_generator import (
    ChapterCandidate,
    ChapterCandidateGenerator,
    ChapterGenerationBatch,
    ChapterGenerationContext,
    ChapterGenerationError,
    ChapterPromptMode,
)
from iread_ai.personalization.generator import (
    PageCandidate,
    PageCandidateRepairer,
    PageGenerationContext,
    PageGenerationError,
)
from iread_ai.personalization.hangul import (
    count_surface_features,
    mask_protected_terms,
    written_syllable_count,
)
from iread_ai.personalization.page_splitter import (
    ChapterPartition,
    DynamicStoryPage,
    PagePartitionError,
    partition_chapter_sentences,
)
from iread_ai.personalization.prompts import (
    BASELINE_PROMPT_MODE,
    PERSONALIZED_PROMPT_MODE,
)
from iread_ai.personalization.repair_policy import (
    apply_repair_batch,
    build_repair_plan,
    child_input_signal_preserved,
    evaluate_repair,
    has_exact_spoken_dialogue,
    has_hard_repair_trigger,
    has_meta_child_reference,
)
from iread_ai.personalization.selector import (
    GenerationProfile,
    SkillPolicy,
)
from iread_ai.personalization.visual_scene import (
    MockVisualScenePlanner,
    VisualSceneGenerationBatch,
    VisualSceneGenerationError,
    VisualScenePlanner,
    build_chapter_visual_scenes,
    load_visual_scene_prompt,
)

logger = logging.getLogger(__name__)

_DIALOGUE_PATTERN = re.compile(r'("[^"\n]+"|“[^”\n]+”|‘[^’\n]+’)')
_HANGUL_PATTERN = re.compile(r"[가-힣]+")
_QUESTION_REWIND_MARKERS = (
    "대신",
    "처음부터",
    "다시 고르",
    "다시 정하",
    "바꿔 볼까요",
    "바꿀까요",
)
_QUESTION_FORWARD_MARKERS = (
    "다음",
    "이제",
    "앞으로",
    "이번에는",
    "그다음",
    "더",
    "깨어나면",
    "잠에서 깬",
    "도착하면",
    "만나면",
    "발견하면",
    "마주치면",
)
_CHILD_CALLBACK_MARKERS = (
    "그 말",
    "그 소리",
    "그 박자",
    "그 리듬",
    "그 구호",
    "그 노래",
    "그 냄새",
    "그 물건",
    "그 도구",
    "그 선택",
    "그 흔적",
    "그 덕분",
    "그 때문에",
    "그 바람에",
)
_RHYTHM_QUESTION_PATTERN = re.compile(r"(?:리듬|박자|흥얼|구호|노래|외칠 말)")
_RHYTHM_BODY_PATTERN = re.compile(r"(?:하나|둘|셋|넷|으쌰|영차|짝짝|쿵짝|리듬|박자)")
_DIALOGUE_FOCUS_PATTERN = re.compile(r"(?:말의 내용|첫마디|한마디|구호|인사|대답|외칠|흥얼|노래)")
_DIALOGUE_QUESTION_PATTERN = re.compile(
    r"(?:뭐라고|무슨 말|어떤.{0,8}말|구호|인사|대답|외칠|흥얼|노래)"
)
_MIXED_ACTION_FOCUS_PATTERN = re.compile(r"(?:행동|방법|작전|길|도구|장소|걸음 모양|고칠 일)")
_GENERIC_CHILD_SIGNAL_WORDS = frozenset(
    {
        "가요",
        "말",
        "말로",
        "말을",
        "봐요",
        "소리",
        "좋아",
        "좋아요",
        "해요",
    }
)
_CHILD_SIGNAL_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "처럼",
    "해요",
    "하고",
    "로",
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "에",
    "요",
)
_ONE_TIME_EVENT_FAMILIES = (
    (
        re.compile(
            r"(?:신호|종|소리|구호).{0,10}(?:울리|울려|들리|들려|외치|외쳐)"
            r"|(?:울리|울려|들리|들려|외치|외쳐).{0,10}"
            r"(?:신호|종|소리|구호)"
        ),
        re.compile(r"(?:신호|종|소리|구호|울릴|외칠)"),
    ),
    (
        re.compile(
            r"(?:경주|여행|게임|놀이|공연).{0,10}(?:시작|출발)"
            r"|(?:시작|출발).{0,10}(?:경주|여행|게임|놀이|공연)"
        ),
        re.compile(r"(?:시작|출발)"),
    ),
    (
        re.compile(r"(?:문|상자|뚜껑|봉투).{0,8}(?:열어|열었|열리|닫아|닫았|닫히)"),
        re.compile(
            r"(?:문|상자|뚜껑|봉투).{0,8}(?:열|닫)"
            r"|(?:무엇|뭐|어떤).{0,8}(?:열|닫)"
        ),
    ),
    (
        re.compile(r"(?:길|도구|물건|방법).{0,8}(?:골라|고르|정해|정했|선택)"),
        re.compile(
            r"(?:길|도구|물건|방법).{0,8}(?:고르|정하|선택)"
            r"|(?:무엇|뭐|어떤|어느).{0,8}(?:고르|정하|선택)"
        ),
    ),
    (
        re.compile(r"(?:결승선|목적지|집|마을).{0,8}(?:도착|닿아|들어가|통과)"),
        re.compile(
            r"(?:결승선|목적지|집|마을).{0,8}(?:도착|닿|들어가|통과)"
            r"|(?:어디|어느 곳).{0,8}(?:도착|닿)"
        ),
    ),
)
_STORY_WORLD_REPLACEMENTS = (
    ("아이의 응원", "힘찬 응원"),
    ("아이의 말", "그 말"),
    ("아이의 답", "그 답"),
    ("아이의 선택", "그 선택"),
    ("독자의 응원", "힘찬 응원"),
    ("사용자의 선택", "그 선택"),
)


@dataclass(slots=True)
class StoryChapterUseCaseError(Exception):
    status_code: int
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class _SkillAssessment:
    code: str
    role: str
    status: str
    occurrences: int | None
    max_occurrences: int | None
    target_min: int | None
    target_max: int | None
    overage: int | None
    target_distance: int | None
    weighted_risk: float


@dataclass(frozen=True, slots=True)
class _PageAssessment:
    page_number: int
    analysis_status: AnalysisStatus
    contract_pass: bool
    contract_failures: tuple[str, ...]
    written_syllable_count: int
    direct_dialogue_count: int
    excluded_overage_count: int
    limited_overage_count: int
    target_distance: int
    risk_per_10: float
    per_skill: tuple[_SkillAssessment, ...]

    @property
    def quality_status(self) -> str:
        if self.analysis_status is not AnalysisStatus.FULL:
            return "ANALYSIS_DEGRADED"
        if not self.contract_pass or self.excluded_overage_count or self.limited_overage_count:
            return "BEST_EFFORT"
        return "PASS"


@dataclass(frozen=True, slots=True)
class _EvaluatedChapter:
    candidate: ChapterCandidate
    partition: ChapterPartition
    pages: tuple[_PageAssessment, ...]
    child_input_reflected: bool
    child_detour_preserved: bool
    child_input_owner_score: int
    child_input_callback_score: int
    question_temporal_penalty: int
    question_answer_leak_penalty: int
    question_focus_penalty: int
    return_score: float

    @property
    def worst_analysis_status(self) -> AnalysisStatus:
        return max(
            (page.analysis_status for page in self.pages),
            key=_analysis_status_rank,
        )

    @property
    def risk_per_10(self) -> float:
        syllables = sum(page.written_syllable_count for page in self.pages)
        weighted_risk = sum(
            page.risk_per_10 * page.written_syllable_count / 10 for page in self.pages
        )
        return 10 * weighted_risk / max(1, syllables)


@dataclass(frozen=True, slots=True)
class _ChapterRepairResult:
    selected: _EvaluatedChapter
    attempted: bool = False
    accepted: bool = False
    elapsed_ms: float = 0.0
    api_call_count: int = 0
    changed_sentences: tuple[dict[str, int], ...] = ()
    reasons: tuple[str, ...] = ()


class PersonalizedStoryChapterService:
    def __init__(
        self,
        *,
        generator: ChapterCandidateGenerator,
        analyzer: KoreanReadingAnalyzer,
        repairer: PageCandidateRepairer | None = None,
        repair_timeout_seconds: float = 6.0,
        candidate_count: int = 3,
        quality_retry_count: int = 0,
        require_contract_pass: bool = False,
        provider_name: str = "mock",
        prompt_mode: ChapterPromptMode = PERSONALIZED_PROMPT_MODE,
        visual_scene_planner: VisualScenePlanner | None = None,
    ) -> None:
        if repair_timeout_seconds <= 0:
            raise ValueError("repair_timeout_seconds must be positive")
        if candidate_count < 1 or candidate_count > 8:
            raise ValueError("candidate_count must be between 1 and 8")
        if quality_retry_count < 0 or quality_retry_count > 2:
            raise ValueError("quality_retry_count must be between 0 and 2")
        if not provider_name.strip():
            raise ValueError("provider_name must not be blank")
        if prompt_mode not in {
            BASELINE_PROMPT_MODE,
            PERSONALIZED_PROMPT_MODE,
        }:
            raise ValueError(f"unsupported chapter prompt mode: {prompt_mode}")
        self._generator = generator
        self._analyzer = analyzer
        self._repairer = repairer
        self._repair_timeout_seconds = repair_timeout_seconds
        self._candidate_count = candidate_count
        self._quality_retry_count = quality_retry_count
        self._require_contract_pass = require_contract_pass
        self._provider_name = provider_name.strip()
        self._prompt_mode = prompt_mode
        self._visual_scene_planner = visual_scene_planner or MockVisualScenePlanner()

    async def generate(
        self,
        request: StoryChapterGenerateRequest,
    ) -> StoryChapterGenerateResponse:
        started = time.perf_counter()
        context = build_chapter_generation_context(request)
        profile = build_chapter_generation_profile(request)

        generation_options: dict[str, Any] = {
            "candidate_count": self._candidate_count,
        }
        generation_profile: GenerationProfile | None = profile
        if self._prompt_mode == BASELINE_PROMPT_MODE:
            generation_options["prompt_mode"] = self._prompt_mode
            generation_profile = None

        generation_ms = 0.0
        pagination_ms = 0.0
        analysis_ms = 0.0
        generation_api_call_count = 0
        best_result: (
            tuple[
                ChapterGenerationBatch,
                _EvaluatedChapter,
                _ChapterRepairResult,
            ]
            | None
        ) = None
        for _attempt in range(self._quality_retry_count + 1):
            try:
                batch = await self._generator.generate(
                    context,
                    generation_profile,
                    **generation_options,
                )
            except ChapterGenerationError as exc:
                raise _provider_error(exc) from exc
            except (TypeError, ValueError) as exc:
                raise StoryChapterUseCaseError(
                    status_code=502,
                    code="MODEL_OUTPUT_INVALID",
                    message="모델이 장 생성 계약에 맞는 결과를 만들지 못했습니다.",
                    retryable=False,
                ) from exc
            generation_ms += float(batch.elapsed_ms)
            generation_api_call_count += 1

            pagination_started = time.perf_counter()
            partitioned: list[tuple[ChapterCandidate, ChapterPartition]] = []
            for candidate in batch.candidates:
                try:
                    partition = _partition_candidate(
                        candidate,
                        request,
                        profile,
                        context,
                    )
                except PagePartitionError:
                    continue
                partitioned.append((candidate, partition))
            pagination_ms += (time.perf_counter() - pagination_started) * 1000
            if not partitioned:
                continue

            analysis_started = time.perf_counter()
            evaluated = await asyncio.to_thread(
                _evaluate_chapters,
                tuple(partitioned),
                context,
                profile,
                self._analyzer,
            )
            analysis_ms += (time.perf_counter() - analysis_started) * 1000
            selected = min(evaluated, key=_chapter_rank)
            repair_result = await _repair_selected_chapter(
                selected=selected,
                request=request,
                context=context,
                profile=profile,
                analyzer=self._analyzer,
                repairer=self._repairer,
                timeout_seconds=self._repair_timeout_seconds,
            )
            selected = repair_result.selected
            current_result = (batch, selected, repair_result)
            if best_result is None or _chapter_rank(selected) < _chapter_rank(best_result[1]):
                best_result = current_result
            if all(page.contract_pass for page in selected.pages):
                best_result = current_result
                break

        if best_result is None:
            raise StoryChapterUseCaseError(
                status_code=502,
                code="MODEL_OUTPUT_INVALID",
                message=("생성된 장을 읽기 분량에 맞는 2~4페이지로 나누지 못했습니다."),
                retryable=False,
            )
        batch, selected, repair_result = best_result
        if self._require_contract_pass and not all(page.contract_pass for page in selected.pages):
            logger.warning(
                json.dumps(
                    {
                        "event": "story_quality_gate_failed",
                        "storyId": request.story_id,
                        "chapterNumber": request.chapter_number,
                        "provider": self._provider_name,
                        "model": batch.model or "unknown",
                        "generationAttemptCount": generation_api_call_count,
                        "quality": _chapter_quality_document(selected.pages),
                        # 계약 위반 원인 추적용: 어떤 문장이 게이트에 걸렸는지 없이는
                        # 반복 실패를 진단할 수 없다.
                        "sentences": [
                            sentence
                            for page in selected.partition.pages
                            for sentence in page.sentences
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            raise StoryChapterUseCaseError(
                status_code=502,
                code="STORY_QUALITY_CONTRACT_FAILED",
                message=(
                    "읽기 품질 기준을 통과하는 이야기를 만들지 "
                    "못했습니다. 잠시 후 다시 시도해 주세요."
                ),
                retryable=False,
            )
        scene_batch, scene_status, scene_fallback_reason = await _generate_visual_scenes(
            planner=self._visual_scene_planner,
            request=request,
            selected=selected,
        )
        total_ms = max(
            (time.perf_counter() - started) * 1000,
            generation_ms,
            analysis_ms,
            pagination_ms,
            repair_result.elapsed_ms,
            scene_batch.elapsed_ms,
        )
        prompt_hash = hashlib.sha256(batch.system_prompt.encode("utf-8")).hexdigest()[:12]
        document = _response_document(
            request=request,
            context=context,
            profile=profile,
            selected=selected,
            model=batch.model or "unknown",
            provider=self._provider_name,
            prompt_hash=prompt_hash,
            candidate_count=len(batch.candidates),
            generation_api_call_count=generation_api_call_count,
            generation_ms=generation_ms,
            analysis_ms=analysis_ms,
            pagination_ms=pagination_ms,
            repair_result=repair_result,
            visual_scene_batch=scene_batch,
            visual_scene_status=scene_status,
            visual_scene_fallback_reason=scene_fallback_reason,
            total_ms=total_ms,
            prompt_mode=self._prompt_mode,
        )
        return StoryChapterGenerateResponse.model_validate(document)


def build_chapter_generation_context(
    request: StoryChapterGenerateRequest,
) -> ChapterGenerationContext:
    state = request.story_state
    beat = request.story_template.current_beat
    child_input = request.branch_input.text if request.branch_input is not None else ""

    def chapter_instruction(text: str) -> str:
        return _materialize_branch_reference(
            _story_world_text(text),
            child_input,
        )

    context_parts = [
        _story_world_text(request.story_template.context),
        f"이번 장의 목표: {chapter_instruction(beat.goal)}",
        (
            "orderedEvents를 순서대로 모두 반영하되, 불필요한 준비나 "
            "반복으로 분량을 늘리지 않습니다."
        ),
    ]
    if beat.allowed_branch_slots:
        context_parts.append("허용된 분기 유형: " + ", ".join(beat.allowed_branch_slots))
    if state.rolling_summary.strip():
        context_parts.append(
            "지금까지의 이야기: " + _story_world_text(state.rolling_summary.strip())
        )
    if state.resolved_facts:
        context_parts.append(
            "이미 일어난 사실: "
            + " / ".join(_story_world_text(fact) for fact in state.resolved_facts)
        )
    if state.unresolved_hooks:
        context_parts.append(
            "이어갈 궁금증: "
            + " / ".join(_story_world_text(hook) for hook in state.unresolved_hooks)
        )
    if state.last_question:
        context_parts.append("직전 질문: " + _story_world_text(state.last_question))

    character_by_id = {character.character_id: character for character in state.characters}
    if character_by_id:
        context_parts.append(
            "등장인물 설정: "
            + " / ".join(
                (f"{character.name}({character.role}; {', '.join(character.immutable_traits)})")
                for character in character_by_id.values()
            )
        )
    required_ids = tuple(
        dict.fromkeys(
            (
                *(
                    character_id
                    for event in request.chapter_plan.ordered_events
                    for character_id in event.required_characters
                ),
                *character_by_id.keys(),
            )
        )
    )
    characters = tuple(character_by_id[character_id].name for character_id in required_ids)
    if not characters:
        characters = tuple(character.name for character in state.characters)
    previous_context = tuple(
        _story_world_text(" ".join(page.sentences)) for page in state.recent_pages[-2:]
    )
    ordered_events = tuple(
        (
            "사건: "
            + chapter_instruction(event.locked_event)
            + (
                ". 반드시 포함할 개념: "
                + ", ".join(chapter_instruction(concept) for concept in event.required_concepts)
                if event.required_concepts
                else ""
            )
        )
        for event in request.chapter_plan.ordered_events
    )
    return ChapterGenerationContext(
        story_title=request.story_template.title,
        story_context="\n".join(context_parts),
        chapter_goal=chapter_instruction(beat.goal),
        ordered_events=ordered_events,
        chapter_number=request.chapter_number,
        child_input=child_input,
        last_question=state.last_question,
        previous_context=previous_context,
        characters=characters,
        question_focus=(
            chapter_instruction(request.chapter_plan.question_focus)
            if request.chapter_plan.question_focus is not None
            else None
        ),
        conclude=request.conclude,
        expected_page_count=request.chapter_plan.max_pages,
        expected_sentences_per_page=(request.generation_profile.content_contract.sentence_count),
    )


def build_chapter_generation_profile(
    request: StoryChapterGenerateRequest,
) -> GenerationProfile:
    payload = request.generation_profile
    contract = payload.content_contract
    protected_terms = tuple(
        dict.fromkeys(
            (
                *payload.protected_terms,
                *(character.name for character in request.story_state.characters),
            )
        )
    )
    from iread_ai.personalization.selector import ContentContract

    return GenerationProfile(
        skills=tuple(
            SkillPolicy(
                code=skill.code,
                role=skill.role,
                max_occurrences=skill.max_occurrences,
                target_min=skill.target_min,
                target_max=skill.target_max,
                unit_penalty=skill.unit_penalty,
            )
            for skill in payload.skills
        ),
        content_contract=ContentContract(
            sentence_count=contract.sentence_count,
            preferred_min_syllables=(contract.preferred_written_syllables.min),
            preferred_max_syllables=(contract.preferred_written_syllables.max),
            accepted_min_syllables=(contract.accepted_written_syllables.min),
            accepted_max_syllables=(contract.accepted_written_syllables.max),
            direct_dialogue=contract.direct_dialogue_count,
        ),
        protected_terms=protected_terms,
    )


def _partition_candidate(
    candidate: ChapterCandidate,
    request: StoryChapterGenerateRequest,
    profile: GenerationProfile,
    context: ChapterGenerationContext,
) -> ChapterPartition:
    options = {
        "min_pages": request.chapter_plan.min_pages,
        "max_pages": request.chapter_plan.max_pages,
        "min_sentences_per_page": 3,
        "max_sentences_per_page": 4,
        "preferred_min_syllables": (profile.content_contract.preferred_min_syllables),
        "preferred_max_syllables": (profile.content_contract.preferred_max_syllables),
        "accepted_min_syllables": (profile.content_contract.accepted_min_syllables),
        "accepted_max_syllables": (profile.content_contract.accepted_max_syllables),
        "direct_dialogue_per_page": (profile.content_contract.direct_dialogue),
    }
    forced_first_break = candidate.child_detour_end_sentence_index if context.child_input else None
    try:
        return partition_chapter_sentences(
            candidate.sentences,
            **options,
            forced_first_break=forced_first_break,
        )
    except PagePartitionError:
        if forced_first_break is None:
            raise
        return partition_chapter_sentences(
            candidate.sentences,
            **options,
        )


async def _repair_selected_chapter(
    *,
    selected: _EvaluatedChapter,
    request: StoryChapterGenerateRequest,
    context: ChapterGenerationContext,
    profile: GenerationProfile,
    analyzer: KoreanReadingAnalyzer,
    repairer: PageCandidateRepairer | None,
    timeout_seconds: float,
) -> _ChapterRepairResult:
    if repairer is None:
        return _ChapterRepairResult(selected=selected)

    hard_failures = {
        "DIRECT_DIALOGUE_COUNT",
        "CURLY_DIALOGUE_FORMAT",
        "META_CHILD_REFERENCE",
        "CHILD_INPUT_NOT_REFLECTED",
    }
    repair_target: (
        tuple[
            DynamicStoryPage,
            PageCandidate,
            PageGenerationContext,
            GenerationProfile,
            Any,
            dict[str, Any],
        ]
        | None
    ) = None
    ranked_pages = sorted(
        zip(selected.partition.pages, selected.pages, strict=True),
        key=lambda row: (
            -sum(failure in hard_failures for failure in row[1].contract_failures),
            -row[1].excluded_overage_count,
            -row[1].limited_overage_count,
            row[0].page_number,
        ),
    )
    for dynamic_page, assessment in ranked_pages:
        if not (
            hard_failures.intersection(assessment.contract_failures)
            or assessment.excluded_overage_count
            or assessment.limited_overage_count
        ):
            continue
        page_candidate = PageCandidate(
            candidate_id=(f"{selected.candidate.candidate_id}-page-{dynamic_page.page_number}"),
            sentences=tuple(dynamic_page.sentences),
        )
        page_profile = replace(
            profile,
            content_contract=replace(
                profile.content_contract,
                sentence_count=len(dynamic_page.sentences),
            ),
        )
        page_context = PageGenerationContext(
            story_title=context.story_title,
            story_context=context.story_context,
            locked_event=context.ordered_events[
                min(
                    dynamic_page.page_number - 1,
                    len(context.ordered_events) - 1,
                )
            ],
            page_number=dynamic_page.page_number,
            child_input=(context.child_input if dynamic_page.page_number == 1 else ""),
            previous_pages=tuple(
                " ".join(page.sentences)
                for page in selected.partition.pages
                if page.page_number < dynamic_page.page_number
            ),
            characters=context.characters,
            required_concepts=(),
            question_focus=(
                context.question_focus
                if dynamic_page.page_number == selected.partition.page_count
                else None
            ),
            conclude=context.conclude,
            expected_sentence_count=len(dynamic_page.sentences),
        )
        try:
            source_evaluation = await asyncio.to_thread(
                evaluate_page_candidate_resilient,
                page_candidate,
                page_profile,
                analyzer,
            )
            repair_plan = await asyncio.to_thread(
                build_repair_plan,
                page_candidate,
                source_evaluation,
                page_profile,
                page_context,
                analyzer,
            )
        except (TypeError, ValueError):
            continue
        if has_hard_repair_trigger(repair_plan) and repair_plan["editable_sentence_indexes"]:
            repair_target = (
                dynamic_page,
                page_candidate,
                page_context,
                page_profile,
                source_evaluation,
                repair_plan,
            )
            break

    if repair_target is None:
        return _ChapterRepairResult(selected=selected)

    (
        target_page,
        source_candidate,
        page_context,
        page_profile,
        source_evaluation,
        repair_plan,
    ) = repair_target
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            repair_batch = await repairer.repair(
                page_context,
                page_profile,
                source_candidate,
                repair_plan=repair_plan,
            )
    except (TimeoutError, PageGenerationError, TypeError, ValueError):
        return _ChapterRepairResult(
            selected=selected,
            attempted=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_call_count=1,
            reasons=("REPAIR_FAILED_SOURCE_RETAINED",),
        )
    if repair_batch.repair_status != "REPAIRED":
        return _ChapterRepairResult(
            selected=selected,
            attempted=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_call_count=1,
            reasons=("MODEL_UNABLE_TO_REPAIR",),
        )

    try:
        proposal_page, changed = apply_repair_batch(
            source_candidate,
            repair_batch,
        )
        proposal_evaluation = await asyncio.to_thread(
            evaluate_page_candidate_resilient,
            proposal_page,
            page_profile,
            analyzer,
        )
        decision = evaluate_repair(
            source_candidate=source_candidate,
            source=source_evaluation,
            proposal_candidate=proposal_page,
            proposal=proposal_evaluation,
            context=page_context,
            profile=page_profile,
            editable_indexes=tuple(
                int(index) for index in repair_plan["editable_sentence_indexes"]
            ),
            changed_sentence_numbers=changed,
        )
        if not decision.accepted:
            return _ChapterRepairResult(
                selected=selected,
                attempted=True,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                api_call_count=1,
                reasons=decision.reasons,
            )

        sentences = list(selected.candidate.sentences)
        global_start = target_page.start_sentence_index - 1
        for local_number in changed:
            sentences[global_start + local_number - 1] = proposal_page.sentences[local_number - 1]
        proposal_candidate = ChapterCandidate(
            candidate_id=selected.candidate.candidate_id,
            sentences=tuple(sentences),
            child_detour_end_sentence_index=(selected.candidate.child_detour_end_sentence_index),
            question=selected.candidate.question,
            subtitle=selected.candidate.subtitle,
            choices=selected.candidate.choices,
        )
        proposal_partition = _partition_candidate(
            proposal_candidate,
            request,
            profile,
            context,
        )
        proposal_chapter = (
            await asyncio.to_thread(
                _evaluate_chapters,
                ((proposal_candidate, proposal_partition),),
                context,
                profile,
                analyzer,
            )
        )[0]
    except (PagePartitionError, TypeError, ValueError):
        return _ChapterRepairResult(
            selected=selected,
            attempted=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_call_count=1,
            reasons=("REPAIRED_CHAPTER_INVALID",),
        )

    if _chapter_rank(proposal_chapter) >= _chapter_rank(selected):
        return _ChapterRepairResult(
            selected=selected,
            attempted=True,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            api_call_count=1,
            reasons=("CHAPTER_QUALITY_NOT_IMPROVED",),
        )
    changed_positions = tuple(
        {
            "globalSentenceNumber": global_start + local_number,
            "pageNumber": target_page.page_number,
            "sentenceNumber": local_number,
        }
        for local_number in changed
    )
    remaining_issue_count = sum(
        len(page.contract_failures) + page.excluded_overage_count + page.limited_overage_count
        for page in proposal_chapter.pages
    )
    decision_reasons = (
        (f"PARTIAL_REPAIR_REMAINING:{remaining_issue_count}",) if remaining_issue_count else ()
    )
    return _ChapterRepairResult(
        selected=proposal_chapter,
        attempted=True,
        accepted=True,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        api_call_count=1,
        changed_sentences=changed_positions,
        reasons=decision_reasons,
    )


def _evaluate_chapters(
    candidates: tuple[tuple[ChapterCandidate, ChapterPartition], ...],
    context: ChapterGenerationContext,
    profile: GenerationProfile,
    analyzer: KoreanReadingAnalyzer,
) -> tuple[_EvaluatedChapter, ...]:
    rows: list[_EvaluatedChapter] = []
    for candidate, partition in candidates:
        page_rows: list[_PageAssessment] = []
        final_page_number = partition.page_count
        chapter_sentences = tuple(
            sentence for page in partition.pages for sentence in page.sentences
        )
        question_temporal_penalty = _question_temporal_penalty(
            candidate.question,
            (
                *context.previous_context,
                *context.ordered_events,
                *candidate.sentences,
            ),
            previous_question=context.last_question,
        )
        question_answer_leak_penalty = _question_answer_leak_penalty(
            candidate.question,
            candidate.choices,
            chapter_sentences,
            context.characters,
        )
        question_focus_penalty = _question_focus_penalty(
            context.question_focus,
            candidate.question,
        )
        (
            child_input_owner_score,
            child_input_callback_score,
        ) = _child_input_causality_scores(context, partition)
        first_page_text = " ".join(partition.pages[0].sentences)
        child_reflected = not context.child_input or _literal_child_input_preserved(
            first_page_text,
            context.child_input,
        )
        for page in partition.pages:
            question = candidate.question if page.page_number == final_page_number else None
            choices = candidate.choices if page.page_number == final_page_number else ()
            page_rows.append(
                _assess_page(
                    page_number=page.page_number,
                    sentences=page.sentences,
                    question=question,
                    choices=choices,
                    profile=profile,
                    analyzer=analyzer,
                    characters=context.characters,
                    question_time_reversal=(
                        page.page_number == final_page_number and question_temporal_penalty > 0
                    ),
                    question_answer_leak=(
                        page.page_number == final_page_number and question_answer_leak_penalty > 0
                    ),
                    child_input_missing=(page.page_number == 1 and not child_reflected),
                    question_focus_mismatch=(
                        page.page_number == final_page_number and question_focus_penalty > 0
                    ),
                )
            )
        rows.append(
            _EvaluatedChapter(
                candidate=candidate,
                partition=partition,
                pages=tuple(page_rows),
                child_input_reflected=child_reflected,
                child_detour_preserved=(
                    not context.child_input
                    or partition.forced_first_break == candidate.child_detour_end_sentence_index
                ),
                child_input_owner_score=child_input_owner_score,
                child_input_callback_score=child_input_callback_score,
                question_temporal_penalty=question_temporal_penalty,
                question_answer_leak_penalty=(question_answer_leak_penalty),
                question_focus_penalty=question_focus_penalty,
                return_score=_return_score(
                    partition.pages[0].sentences[-1],
                    context.ordered_events[0],
                ),
            )
        )
    return tuple(rows)


def _assess_page(
    *,
    page_number: int,
    sentences: tuple[str, ...],
    question: str | None,
    choices: tuple[str, ...],
    profile: GenerationProfile,
    analyzer: KoreanReadingAnalyzer,
    characters: tuple[str, ...],
    question_time_reversal: bool = False,
    question_answer_leak: bool = False,
    child_input_missing: bool = False,
    question_focus_mismatch: bool = False,
) -> _PageAssessment:
    body = _analyze_resilient(
        sentences,
        profile.protected_terms,
        analyzer,
    )
    scopes: list[tuple[CandidateAnalysis, float]] = [(body, 1.0)]
    if question is not None:
        scopes.append(
            (
                _analyze_resilient(
                    (question,),
                    profile.protected_terms,
                    analyzer,
                ),
                0.8,
            )
        )
        scopes.append(
            (
                _analyze_resilient(
                    choices,
                    profile.protected_terms,
                    analyzer,
                ),
                0.5,
            )
        )

    contract = profile.content_contract
    failures: list[str] = []
    if not 3 <= len(sentences) <= 4:
        failures.append("SENTENCE_COUNT")
    if not (
        contract.accepted_min_syllables <= body.written_syllables <= contract.accepted_max_syllables
    ):
        failures.append("WRITTEN_SYLLABLE_RANGE")
    if body.dialogue_sentence_count > contract.direct_dialogue:
        failures.append("DIRECT_DIALOGUE_COUNT")
    if body.dialogue_sentence_count > 0 and not has_exact_spoken_dialogue(sentences, characters):
        failures.append("CURLY_DIALOGUE_FORMAT")
    if has_meta_child_reference(sentences, characters):
        failures.append("META_CHILD_REFERENCE")
    if question_time_reversal:
        failures.append("QUESTION_TIME_REVERSAL")
    if question_answer_leak:
        failures.append("QUESTION_ANSWER_ALREADY_IN_BODY")
    if child_input_missing:
        failures.append("CHILD_INPUT_NOT_REFLECTED")
    if question_focus_mismatch:
        failures.append("QUESTION_FOCUS_MISMATCH")

    excluded_overage = 0
    limited_overage = 0
    target_distance_total = 0
    per_skill: list[_SkillAssessment] = []
    total_risk = 0.0
    for skill in profile.skills:
        scope_counts = tuple(
            (_analysis_occurrences(analysis, skill.code), weight) for analysis, weight in scopes
        )
        raw_count = sum(count for count, _ in scope_counts)
        weighted_count = sum(count * weight for count, weight in scope_counts)
        unverified = skill.code.startswith("PHONO_") and any(
            analysis.status is not AnalysisStatus.FULL for analysis, _ in scopes
        )
        occurrences = None if unverified else raw_count
        status = "UNVERIFIED" if unverified else "PASS"
        overage: int | None = None
        target_distance: int | None = None
        weighted_risk = 0.0
        if occurrences is not None and skill.role in {"EXCLUDED", "LIMITED"}:
            overage = max(
                0,
                occurrences - (skill.max_occurrences or 0),
            )
            weighted_risk = weighted_count * skill.unit_penalty
            if skill.role == "EXCLUDED":
                excluded_overage += overage
            else:
                limited_overage += overage
            if overage:
                status = "OVER_LIMIT"
        elif occurrences is not None and skill.role == "TARGET":
            minimum = skill.target_min if skill.target_min is not None else occurrences
            maximum = skill.target_max if skill.target_max is not None else occurrences
            target_distance = _range_distance(
                occurrences,
                minimum,
                maximum,
            )
            target_distance_total += target_distance
            weighted_risk = target_distance * skill.unit_penalty
            if target_distance:
                status = "OUTSIDE_TARGET"
        total_risk += weighted_risk
        per_skill.append(
            _SkillAssessment(
                code=skill.code,
                role=skill.role,
                status=status,
                occurrences=occurrences,
                max_occurrences=skill.max_occurrences,
                target_min=skill.target_min,
                target_max=skill.target_max,
                overage=overage,
                target_distance=target_distance,
                weighted_risk=weighted_risk,
            )
        )

    weighted_syllables = sum(analysis.written_syllables * weight for analysis, weight in scopes)
    analysis_status = max(
        (analysis.status for analysis, _ in scopes),
        key=_analysis_status_rank,
    )
    return _PageAssessment(
        page_number=page_number,
        analysis_status=analysis_status,
        contract_pass=not failures,
        contract_failures=tuple(failures),
        written_syllable_count=body.written_syllables,
        direct_dialogue_count=body.dialogue_sentence_count,
        excluded_overage_count=excluded_overage,
        limited_overage_count=limited_overage,
        target_distance=target_distance_total,
        risk_per_10=10 * total_risk / max(1.0, weighted_syllables),
        per_skill=tuple(per_skill),
    )


def _analyze_resilient(
    texts: tuple[str, ...],
    protected_terms: tuple[str, ...],
    analyzer: KoreanReadingAnalyzer,
) -> CandidateAnalysis:
    try:
        return analyzer.analyze(
            texts,
            protected_terms=protected_terms,
        )
    except Exception:
        return _surface_only_analysis(texts, protected_terms)


def _surface_only_analysis(
    texts: tuple[str, ...],
    protected_terms: tuple[str, ...],
) -> CandidateAnalysis:
    joined = " ".join(texts)
    surface = count_surface_features(joined)
    controllable = count_surface_features(mask_protected_terms(joined, protected_terms))
    protected = {
        code: count - controllable.get(code, 0)
        for code, count in surface.items()
        if count - controllable.get(code, 0) > 0
    }
    return CandidateAnalysis(
        status=AnalysisStatus.SURFACE_ONLY,
        surface_feature_counts=surface,
        controllable_surface_feature_counts=controllable,
        protected_surface_feature_counts=protected,
        phonological_rule_counts={},
        written_syllables=written_syllable_count(joined),
        dialogue_sentence_count=sum(
            _DIALOGUE_PATTERN.search(sentence) is not None for sentence in texts
        ),
        pronunciations=(),
        kiwi_token_count=0,
        g2p_review_sentence_count=len(texts),
        latency_ms=0.0,
        error="local linguistic analysis unavailable",
    )


def _analysis_occurrences(
    analysis: CandidateAnalysis,
    code: str,
) -> int:
    if code.startswith("PHONO_"):
        return int(analysis.phonological_rule_counts.get(code, 0))
    return int(analysis.controllable_surface_feature_counts.get(code, 0))


def _chapter_rank(row: _EvaluatedChapter) -> tuple[Any, ...]:
    return (
        not row.child_input_reflected,
        not row.child_detour_preserved,
        row.question_temporal_penalty,
        row.question_answer_leak_penalty,
        row.question_focus_penalty,
        sum(not page.contract_pass for page in row.pages),
        sum(len(page.contract_failures) for page in row.pages),
        max(page.excluded_overage_count for page in row.pages),
        sum(page.excluded_overage_count for page in row.pages),
        -row.child_input_owner_score,
        -row.child_input_callback_score,
        sum(page.limited_overage_count for page in row.pages),
        sum(page.target_distance for page in row.pages),
        _analysis_status_rank(row.worst_analysis_status),
        -row.return_score,
        row.risk_per_10,
        row.partition.contract_penalty,
        row.partition.page_count,
        row.candidate.candidate_id,
    )


def _child_input_causality_scores(
    context: ChapterGenerationContext,
    partition: ChapterPartition,
) -> tuple[int, int]:
    child_input = context.child_input.strip()
    if not child_input:
        return 2, 2

    first_page = partition.pages[0].sentences
    signal_indices = tuple(
        index
        for index, sentence in enumerate(first_page)
        if _literal_child_input_preserved(sentence, child_input)
    )
    target_characters = tuple(
        character
        for character in context.characters
        if context.last_question is not None and character in context.last_question
    )
    if not signal_indices:
        owner_score = 0
    elif not target_characters:
        owner_score = 1
    elif any(
        any(character in first_page[index] for character in target_characters)
        for index in signal_indices
    ):
        owner_score = 2
    elif any(
        any(character in first_page[nearby_index] for character in target_characters)
        for index in signal_indices
        for nearby_index in range(
            max(0, index - 1),
            min(len(first_page), index + 2),
        )
    ):
        owner_score = 1
    else:
        owner_score = 0

    later_text = " ".join(sentence for page in partition.pages[1:] for sentence in page.sentences)
    if _literal_child_input_preserved(later_text, child_input):
        callback_score = 2
    elif any(marker in later_text for marker in _CHILD_CALLBACK_MARKERS):
        callback_score = 1
    else:
        callback_score = 0
    return owner_score, callback_score


def _literal_child_input_preserved(
    sentence: str,
    child_input: str,
) -> bool:
    sentence_hangul = "".join(_HANGUL_PATTERN.findall(sentence))
    terms: list[str] = []
    for raw_word in _HANGUL_PATTERN.findall(child_input):
        word = raw_word
        for suffix in _CHILD_SIGNAL_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                word = word[: -len(suffix)]
                break
        if len(word) >= 2 and word not in _GENERIC_CHILD_SIGNAL_WORDS:
            terms.append(word)
    if not terms:
        return child_input_signal_preserved(sentence, child_input)
    matched = sum(term in sentence_hangul for term in terms)
    return matched >= (len(terms) + 1) // 2


def _question_answer_leak_penalty(
    question: str | None,
    choices: tuple[str, ...],
    completed_sentences: tuple[str, ...],
    characters: tuple[str, ...],
) -> int:
    if question is None:
        return 0

    penalty = 0
    for choice in choices:
        choice_bigrams = _hangul_bigrams(choice)
        if len(choice_bigrams) < 3:
            continue
        coverage = max(
            (
                len(choice_bigrams & _hangul_bigrams(sentence)) / len(choice_bigrams)
                for sentence in completed_sentences
            ),
            default=0.0,
        )
        if coverage >= 0.6:
            penalty += 1

    if _RHYTHM_QUESTION_PATTERN.search(question) is not None:
        target_characters = tuple(character for character in characters if character in question)
        if any(
            _DIALOGUE_PATTERN.search(sentence) is not None
            and _RHYTHM_BODY_PATTERN.search(sentence) is not None
            and (
                not target_characters
                or any(character in sentence for character in target_characters)
            )
            for sentence in completed_sentences
        ):
            penalty += 2
    return penalty


def _question_focus_penalty(
    question_focus: str | None,
    question: str | None,
) -> int:
    if question_focus is None or question is None:
        return 0
    dialogue_only = (
        _DIALOGUE_FOCUS_PATTERN.search(question_focus) is not None
        and _MIXED_ACTION_FOCUS_PATTERN.search(question_focus) is None
    )
    if dialogue_only and _DIALOGUE_QUESTION_PATTERN.search(question) is None:
        return 2
    return 0


def _question_temporal_penalty(
    question: str | None,
    completed_texts: tuple[str, ...],
    *,
    previous_question: str | None = None,
) -> int:
    if question is None:
        return 0
    normalized_question = " ".join(question.split())
    if previous_question is not None:
        normalized_previous = re.sub(
            r"[\s.?!。？！]+",
            "",
            previous_question,
        )
        if re.sub(r"[\s.?!。？！]+", "", normalized_question) == (normalized_previous):
            return 3
    completed_text = " ".join(completed_texts)
    completed_families = {
        index
        for index, (completed_pattern, _) in enumerate(_ONE_TIME_EVENT_FAMILIES)
        if completed_pattern.search(completed_text)
    }
    repeated_families = {
        index
        for index, (_, question_pattern) in enumerate(_ONE_TIME_EVENT_FAMILIES)
        if question_pattern.search(normalized_question)
    }
    if not completed_families.intersection(repeated_families):
        return 0
    if any(marker in normalized_question for marker in _QUESTION_REWIND_MARKERS):
        return 2
    if any(marker in normalized_question for marker in _QUESTION_FORWARD_MARKERS):
        return 0
    return 1


async def _generate_visual_scenes(
    *,
    planner: VisualScenePlanner,
    request: StoryChapterGenerateRequest,
    selected: _EvaluatedChapter,
) -> tuple[VisualSceneGenerationBatch, str, str | None]:
    question = selected.candidate.question if not request.conclude else None
    choices = selected.candidate.choices if not request.conclude else ()
    started = time.perf_counter()
    try:
        batch = await planner.generate(
            request=request,
            pages=selected.partition.pages,
            question=question,
            choices=choices,
        )
    except VisualSceneGenerationError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if "timed out" in str(exc).lower():
            reason = "TIMEOUT"
        elif exc.raw_output is not None or "invalid" in str(exc).lower():
            reason = "INVALID_OUTPUT"
        else:
            reason = "UPSTREAM_FAILURE"
        fallback_scenes = build_chapter_visual_scenes(
            request=request,
            pages=selected.partition.pages,
        )
        return (
            VisualSceneGenerationBatch(
                scenes=fallback_scenes,
                raw_output="",
                elapsed_ms=elapsed_ms,
                model="deterministic",
                system_prompt=load_visual_scene_prompt(),
                user_prompt="",
                api_call_count=1,
            ),
            "DETERMINISTIC_FALLBACK",
            reason,
        )
    status = "LLM_GENERATED" if batch.api_call_count > 0 else "MOCK"
    return batch, status, None


def _response_document(
    *,
    request: StoryChapterGenerateRequest,
    context: ChapterGenerationContext,
    profile: GenerationProfile,
    selected: _EvaluatedChapter,
    model: str,
    provider: str,
    prompt_hash: str,
    candidate_count: int,
    generation_api_call_count: int,
    generation_ms: float,
    analysis_ms: float,
    pagination_ms: float,
    repair_result: _ChapterRepairResult,
    visual_scene_batch: VisualSceneGenerationBatch,
    visual_scene_status: str,
    visual_scene_fallback_reason: str | None,
    total_ms: float,
    prompt_mode: ChapterPromptMode,
) -> dict[str, Any]:
    final_page_number = selected.partition.page_count
    visual_scenes = visual_scene_batch.scenes
    pages: list[dict[str, Any]] = []
    quality_pages: list[dict[str, Any]] = []
    for dynamic_page, assessment, visual_scene in zip(
        selected.partition.pages,
        selected.pages,
        visual_scenes,
        strict=True,
    ):
        is_final = dynamic_page.page_number == final_page_number
        branch_required = is_final and not request.conclude
        pages.append(
            {
                "pageNumber": dynamic_page.page_number,
                "sentences": list(dynamic_page.sentences),
                "visualScene": visual_scene,
                "question": (selected.candidate.question if branch_required else None),
                "subtitle": (_candidate_subtitle(selected.candidate) if branch_required else None),
                "choices": (list(selected.candidate.choices) if branch_required else []),
                "requiresBranchInput": branch_required,
            }
        )
        quality_pages.append(
            {
                "pageNumber": assessment.page_number,
                "quality": _page_quality_document(assessment),
            }
        )

    final_question = selected.candidate.question if not request.conclude else None
    page_text = " ".join(
        sentence for page in selected.partition.pages for sentence in page.sentences
    )
    rolling_summary = " ".join(
        part
        for part in (
            request.story_state.rolling_summary.strip(),
            page_text,
        )
        if part
    )[-4000:]
    chapter_quality = _chapter_quality_document(selected.pages)
    visual_scene_prompt_hash = hashlib.sha256(
        visual_scene_batch.system_prompt.encode("utf-8")
    ).hexdigest()[:12]
    removed_hooks = (
        [request.story_state.last_question]
        if request.branch_input is not None and request.story_state.last_question is not None
        else []
    )
    return {
        "requestId": request.request_id,
        "schemaVersion": 3,
        "generationId": f"chapter-{uuid.uuid4().hex}",
        "storyId": request.story_id,
        "storyRevision": request.story_revision,
        "chapterNumber": request.chapter_number,
        "pages": pages,
        "quality": {
            "chapter": chapter_quality,
            "pages": quality_pages,
        },
        "generation": {
            "provider": provider,
            "model": model,
            "promptVersion": f"chapter-{prompt_mode}-{prompt_hash}",
            "generationProfileVersion": (request.generation_profile.generation_profile_version),
            "policyHash": request.generation_profile.policy_hash,
            "candidateCount": candidate_count,
            "selectedCandidateId": selected.candidate.candidate_id,
            "pageCount": final_page_number,
            "apiCallCount": (
                generation_api_call_count
                + repair_result.api_call_count
                + visual_scene_batch.api_call_count
            ),
            "repairAttempted": repair_result.attempted,
            "repairAccepted": repair_result.accepted,
            "changedSentences": list(repair_result.changed_sentences),
            "repairDecisionReasons": list(repair_result.reasons),
            "visualSceneStatus": visual_scene_status,
            "visualSceneModel": visual_scene_batch.model,
            "visualScenePromptVersion": (f"visual-scene-{visual_scene_prompt_hash}"),
            "visualSceneFallbackReason": visual_scene_fallback_reason,
        },
        "timingMs": {
            "generation": generation_ms,
            "analysis": analysis_ms,
            "pagination": pagination_ms,
            "repair": repair_result.elapsed_ms,
            "visualScene": visual_scene_batch.elapsed_ms,
            "total": total_ms,
        },
        "statePatch": {
            "expectedBaseRevision": request.story_revision,
            "rollingSummary": rolling_summary,
            "resolvedFactsAdded": [
                event.locked_event for event in request.chapter_plan.ordered_events
            ],
            "unresolvedHooksAdded": ([final_question] if final_question is not None else []),
            "unresolvedHooksRemoved": removed_hooks,
            "charactersUpserted": [],
            "lastQuestion": final_question,
        },
    }


def _candidate_subtitle(candidate: ChapterCandidate) -> str:
    if candidate.subtitle is not None and candidate.subtitle.strip():
        return candidate.subtitle.strip()[:40]
    source = next(
        (choice.strip() for choice in candidate.choices if choice.strip()),
        (candidate.question or "다음 이야기").strip(),
    )
    return source[:40]


def _page_quality_document(
    assessment: _PageAssessment,
) -> dict[str, Any]:
    return {
        "status": assessment.quality_status,
        "analysisStatus": assessment.analysis_status.value,
        "contractPass": assessment.contract_pass,
        "contractFailures": list(assessment.contract_failures),
        "writtenSyllableCount": assessment.written_syllable_count,
        "directDialogueCount": assessment.direct_dialogue_count,
        "excludedOverageCount": assessment.excluded_overage_count,
        "limitedOverageCount": assessment.limited_overage_count,
        "riskPer10": float(assessment.risk_per_10),
        "perSkill": [
            {
                "code": skill.code,
                "role": skill.role,
                "status": skill.status,
                "occurrences": skill.occurrences,
                "maxOccurrences": skill.max_occurrences,
                "targetMin": skill.target_min,
                "targetMax": skill.target_max,
                "overage": skill.overage,
                "targetDistance": skill.target_distance,
                "weightedRisk": float(skill.weighted_risk),
            }
            for skill in assessment.per_skill
        ],
    }


def _chapter_quality_document(
    pages: tuple[_PageAssessment, ...],
) -> dict[str, Any]:
    written_syllables = sum(page.written_syllable_count for page in pages)
    direct_dialogue = sum(page.direct_dialogue_count for page in pages)
    excluded_overage = sum(page.excluded_overage_count for page in pages)
    limited_overage = sum(page.limited_overage_count for page in pages)
    risk_per_10 = sum(page.risk_per_10 * page.written_syllable_count for page in pages) / max(
        1, written_syllables
    )
    analysis_status = max(
        (page.analysis_status for page in pages),
        key=_analysis_status_rank,
    )
    status = (
        "ANALYSIS_DEGRADED"
        if analysis_status is not AnalysisStatus.FULL
        else "BEST_EFFORT"
        if any(page.quality_status != "PASS" for page in pages)
        else "PASS"
    )
    failures = [
        f"P{page.page_number}_{failure}" for page in pages for failure in page.contract_failures
    ][:20]
    return {
        "status": status,
        "analysisStatus": analysis_status.value,
        "contractPass": all(page.contract_pass for page in pages),
        "contractFailures": failures,
        "writtenSyllableCount": written_syllables,
        "directDialogueCount": direct_dialogue,
        "excludedOverageCount": excluded_overage,
        "limitedOverageCount": limited_overage,
        "riskPer10": float(risk_per_10),
        "perSkill": _aggregate_skill_quality(pages),
    }


def _aggregate_skill_quality(
    pages: tuple[_PageAssessment, ...],
) -> list[dict[str, Any]]:
    if not pages:
        return []
    rows: list[dict[str, Any]] = []
    for skill_index in range(len(pages[0].per_skill)):
        skills = [page.per_skill[skill_index] for page in pages]
        first = skills[0]
        unverified = any(skill.occurrences is None for skill in skills)
        occurrences = None if unverified else sum(int(skill.occurrences or 0) for skill in skills)
        overage = None if unverified else sum(int(skill.overage or 0) for skill in skills)
        target_distance = (
            None if unverified else sum(int(skill.target_distance or 0) for skill in skills)
        )
        status = "UNVERIFIED" if unverified else "PASS"
        if not unverified and overage:
            status = "OVER_LIMIT"
        elif not unverified and target_distance:
            status = "OUTSIDE_TARGET"
        rows.append(
            {
                "code": first.code,
                "role": first.role,
                "status": status,
                "occurrences": occurrences,
                "maxOccurrences": first.max_occurrences,
                "targetMin": first.target_min,
                "targetMax": first.target_max,
                "overage": overage,
                "targetDistance": target_distance,
                "weightedRisk": float(sum(skill.weighted_risk for skill in skills)),
            }
        )
    return rows


def _provider_error(
    exc: ChapterGenerationError,
) -> StoryChapterUseCaseError:
    if "timed out" in str(exc).lower():
        return StoryChapterUseCaseError(
            status_code=504,
            code="MODEL_TIMEOUT",
            message="장 생성 시간이 제한을 초과했습니다.",
            retryable=True,
        )
    if exc.raw_output is not None or "invalid" in str(exc).lower():
        return StoryChapterUseCaseError(
            status_code=502,
            code="MODEL_OUTPUT_INVALID",
            message="모델이 장 생성 계약에 맞는 결과를 만들지 못했습니다.",
            retryable=False,
        )
    return StoryChapterUseCaseError(
        status_code=502,
        code="MODEL_UPSTREAM_FAILURE",
        message="장 생성 모델에 연결하지 못했습니다.",
        retryable=exc.retryable,
    )


def _story_world_text(text: str) -> str:
    result = text
    for source, replacement in _STORY_WORLD_REPLACEMENTS:
        result = result.replace(source, replacement)
    return result


def _materialize_branch_reference(text: str, child_input: str) -> str:
    answer = child_input.strip()
    if not answer:
        return text
    quoted = f"“{answer}”"
    replacements = (
        ("직전 답의", f"확정된 답 {quoted}에 담긴"),
        ("직전 답으로", f"확정된 답 {quoted}대로"),
        ("직전 답을", f"확정된 답 {quoted} 내용을"),
        ("직전 답에서", f"확정된 답 {quoted}에서"),
        ("직전 답", f"확정된 답 {quoted}"),
        ("직전 선택대로", f"확정된 선택 {quoted}대로"),
        ("직전 선택으로", f"확정된 선택 {quoted}에 따라"),
        ("직전 선택을", f"확정된 선택 {quoted} 내용을"),
        ("직전 선택", f"확정된 선택 {quoted}"),
        ("직전 방법으로", f"확정된 방법 {quoted}대로"),
        ("직전 방법", f"확정된 방법 {quoted}"),
        ("직전 대화", f"직전에 고른 말 {quoted}"),
    )
    result = text
    for source, replacement in replacements:
        result = result.replace(source, replacement)
    return result


def _analysis_status_rank(status: AnalysisStatus) -> int:
    return {
        AnalysisStatus.FULL: 0,
        AnalysisStatus.UNRELIABLE: 1,
        AnalysisStatus.SURFACE_ONLY: 2,
    }[status]


def _range_distance(
    value: int,
    minimum: int,
    maximum: int,
) -> int:
    if value < minimum:
        return minimum - value
    if value > maximum:
        return value - maximum
    return 0


def _return_score(actual: str, expected: str) -> float:
    expected_bigrams = _hangul_bigrams(expected)
    if not expected_bigrams:
        return 0.0
    actual_bigrams = _hangul_bigrams(actual)
    return len(expected_bigrams & actual_bigrams) / len(expected_bigrams)


def _hangul_bigrams(text: str) -> set[str]:
    hangul = "".join(_HANGUL_PATTERN.findall(text))
    if len(hangul) < 2:
        return {hangul} if hangul else set()
    return {hangul[index : index + 2] for index in range(len(hangul) - 1)}


__all__ = [
    "PersonalizedStoryChapterService",
    "StoryChapterUseCaseError",
    "build_chapter_generation_context",
    "build_chapter_generation_profile",
]
