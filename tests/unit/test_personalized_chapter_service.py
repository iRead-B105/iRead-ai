from __future__ import annotations

import re
from typing import Any

import pytest

from iread_ai.application.personalized_chapter_service import (
    PersonalizedStoryChapterService,
    StoryChapterUseCaseError,
    build_chapter_generation_context,
)
from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.personalization.analyzer import AnalysisStatus, CandidateAnalysis
from iread_ai.personalization.chapter_generator import (
    ChapterCandidate,
    ChapterGenerationBatch,
    ChapterGenerationContext,
)
from iread_ai.personalization.generator import (
    PageCandidate,
    PageGenerationContext,
    RepairBatch,
    RepairReplacement,
)
from iread_ai.personalization.hangul import written_syllable_count
from iread_ai.personalization.page_splitter import DynamicStoryPage
from iread_ai.personalization.selector import GenerationProfile
from iread_ai.personalization.visual_scene import (
    VisualSceneGenerationBatch,
    VisualSceneGenerationError,
    build_chapter_visual_scenes,
)
from tests.unit.test_story_chapter_contracts import request_payload

_DIALOGUE = re.compile(r'("[^"\n]+"|“[^”\n]+”|‘[^’\n]+’)')


def chapter_candidate(
    *,
    candidate_id: str = "chapter-candidate-1",
    child_input: bool = True,
) -> ChapterCandidate:
    return ChapterCandidate(
        candidate_id=candidate_id,
        sentences=(
            "숲길에 방구 소리가 크게 울려요.",
            "토끼가 놀라 앞으로 폴짝 뛰어요.",
            "거북이가 “누구 소리야?” 하고 물어요.",
            "둘은 다시 경주 길을 따라가요.",
            "토끼는 나무 아래서 잠깐 쉬어요.",
            "거북이는 멈추지 않고 언덕을 올라요.",
            "“곧 따라갈게!” 토끼가 크게 외쳐요.",
            "두 친구 앞에 반짝이는 표지판이 나타나요.",
        ),
        child_detour_end_sentence_index=4 if child_input else None,
        question="표지판은 어느 길을 가리킬까요?",
        choices=("꽃길로 가요.", "돌길로 가요.", "숲길로 가요."),
    )


class RecordingChapterGenerator:
    def __init__(self, candidate: ChapterCandidate) -> None:
        self.candidate = candidate
        self.calls: list[tuple[ChapterGenerationContext, GenerationProfile, int]] = []

    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: GenerationProfile,
        *,
        candidate_count: int = 3,
    ) -> ChapterGenerationBatch:
        self.calls.append((context, profile, candidate_count))
        candidates = tuple(
            ChapterCandidate(
                candidate_id=f"chapter-candidate-{index}",
                sentences=self.candidate.sentences,
                child_detour_end_sentence_index=(self.candidate.child_detour_end_sentence_index),
                question=self.candidate.question,
                choices=self.candidate.choices,
            )
            for index in range(1, candidate_count + 1)
        )
        return ChapterGenerationBatch(
            candidates=candidates,
            raw_output="RAW_OUTPUT_MUST_NOT_ESCAPE",
            elapsed_ms=120.0,
            model="test-chapter-model",
            system_prompt="chapter prompt",
            user_prompt="STUDENT_PRIVATE_DATA",
        )


class RecordingVisualScenePlanner:
    def __init__(self) -> None:
        self.pages: tuple[DynamicStoryPage, ...] = ()

    async def generate(
        self,
        *,
        request: StoryChapterGenerateRequest,
        pages: tuple[DynamicStoryPage, ...],
        question: str | None,
        choices: tuple[str, ...],
    ) -> VisualSceneGenerationBatch:
        del question, choices
        self.pages = tuple(pages)
        return VisualSceneGenerationBatch(
            scenes=build_chapter_visual_scenes(
                request=request,
                pages=pages,
            ),
            raw_output="scene-json",
            elapsed_ms=75.0,
            model="scene-model",
            system_prompt="scene prompt",
            user_prompt="scene input",
            api_call_count=1,
        )


class FailingVisualScenePlanner:
    async def generate(
        self,
        *,
        request: StoryChapterGenerateRequest,
        pages: tuple[DynamicStoryPage, ...],
        question: str | None,
        choices: tuple[str, ...],
    ) -> VisualSceneGenerationBatch:
        del request, pages, question, choices
        raise VisualSceneGenerationError(
            "visual-scene generation timed out",
            retryable=True,
        )


class FixedChapterBatchGenerator:
    def __init__(self, candidates: tuple[ChapterCandidate, ...]) -> None:
        self.candidates = candidates

    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: GenerationProfile,
        *,
        candidate_count: int = 3,
    ) -> ChapterGenerationBatch:
        del context, profile
        assert candidate_count == len(self.candidates)
        return ChapterGenerationBatch(
            candidates=self.candidates,
            raw_output="fixed candidates",
            elapsed_ms=20.0,
            model="test-chapter-model",
            system_prompt="chapter prompt",
            user_prompt="test context",
        )


class SequencedChapterGenerator:
    def __init__(self, candidates: tuple[ChapterCandidate, ...]) -> None:
        self.candidates = candidates
        self.calls = 0

    async def generate(
        self,
        context: ChapterGenerationContext,
        profile: GenerationProfile,
        *,
        candidate_count: int = 1,
    ) -> ChapterGenerationBatch:
        del context, profile
        assert candidate_count == 1
        index = min(self.calls, len(self.candidates) - 1)
        self.calls += 1
        candidate = self.candidates[index]
        return ChapterGenerationBatch(
            candidates=(candidate,),
            raw_output=f"attempt-{self.calls}",
            elapsed_ms=20.0,
            model="gms-test-model",
            system_prompt="chapter prompt",
            user_prompt="test context",
        )


class DeterministicAnalyzer:
    def analyze(
        self,
        sentences: tuple[str, ...],
        protected_terms: tuple[str, ...] = (),
    ) -> CandidateAnalysis:
        del protected_terms
        text = " ".join(sentences)
        return CandidateAnalysis(
            status=AnalysisStatus.FULL,
            surface_feature_counts={},
            controllable_surface_feature_counts={},
            protected_surface_feature_counts={},
            phonological_rule_counts={},
            written_syllables=written_syllable_count(text),
            dialogue_sentence_count=sum(
                _DIALOGUE.search(sentence) is not None for sentence in sentences
            ),
            pronunciations=(),
            kiwi_token_count=len(sentences),
            g2p_review_sentence_count=0,
            latency_ms=0.0,
        )


class DialogueRepairer:
    def __init__(self) -> None:
        self.calls = 0

    async def repair(
        self,
        context: PageGenerationContext,
        profile: GenerationProfile,
        source_candidate: PageCandidate,
        *,
        repair_plan: dict[str, Any],
    ) -> RepairBatch:
        del context, profile, repair_plan
        self.calls += 1
        replacement = RepairReplacement(
            sentence_index=2,
            sentence="거북이가 “난 멈추지 않고 갈 거야!” 하고 말해요.",
        )
        return RepairBatch(
            source_candidate_id=source_candidate.candidate_id,
            repair_status="REPAIRED",
            replacements=(replacement,),
            raw_output="repair",
            elapsed_ms=10.0,
            model="test-repair",
        )


class RecordingDynamicPageRepairer:
    def __init__(self) -> None:
        self.calls: list[tuple[PageGenerationContext, GenerationProfile, PageCandidate]] = []

    async def repair(
        self,
        context: PageGenerationContext,
        profile: GenerationProfile,
        source_candidate: PageCandidate,
        *,
        repair_plan: dict[str, Any],
    ) -> RepairBatch:
        del repair_plan
        self.calls.append((context, profile, source_candidate))
        return RepairBatch(
            source_candidate_id=source_candidate.candidate_id,
            repair_status="REPAIRED",
            replacements=(
                RepairReplacement(
                    sentence_index=2,
                    sentence=("거북이는 “멈추지 않고 언덕을 천천히 갈 거야.” 하고 말해요."),
                ),
            ),
            raw_output="repair",
            elapsed_ms=10.0,
            model="test-repair",
        )


def _request() -> StoryChapterGenerateRequest:
    payload = request_payload(min_pages=2, max_pages=4)
    payload["branchInput"] = {
        "source": "TEXT_CONFIRMED",
        "text": "방구 소리",
    }
    payload["storyTemplate"]["title"] = "토끼와 거북이"
    payload["storyTemplate"]["context"] = "토끼와 거북이가 숲길에서 경주해요."
    payload["storyTemplate"]["currentBeat"]["goal"] = "짧은 소동 뒤 경주가 다시 이어져요."
    payload["storyState"]["characters"][0]["name"] = "토끼"
    payload["storyState"]["characters"][1]["name"] = "거북이"
    payload["chapterPlan"]["orderedEvents"][0]["lockedEvent"] = (
        "아이의 소리가 숲길에 실제로 울려요."
    )
    payload["chapterPlan"]["orderedEvents"][1]["lockedEvent"] = "두 친구가 경주로 돌아가요."
    payload["chapterPlan"]["questionFocus"] = "다음에 갈 길"
    return StoryChapterGenerateRequest.model_validate(payload)


def test_visual_scene_infers_restrained_character_emotions() -> None:
    page = DynamicStoryPage(
        page_number=1,
        start_sentence_index=1,
        end_sentence_index=4,
        sentences=(
            "토끼가 코를 세우며 거북이를 쳐다봐요.",
            "“난 훨씬 빨라!” 토끼가 도발해요.",
            "거북이는 천천히라도 끝까지 가겠다고 말해요.",
            "두 친구는 같은 출발선으로 가요.",
        ),
        written_syllable_count=55,
        direct_dialogue_count=1,
        contract_pass=True,
        contract_failures=(),
        accepted_length_distance=0,
        preferred_length_distance=0,
    )

    scene = build_chapter_visual_scenes(
        request=_request(),
        pages=(page,),
    )[0]
    emotions = {
        character["characterId"]: character["emotion"]["type"] for character in scene["characters"]
    }

    assert emotions == {
        "hare": "CONFIDENT",
        "tortoise": "FOCUSED",
    }


async def test_generates_one_dynamic_chapter_and_splits_it_into_two_pages() -> None:
    generator = RecordingChapterGenerator(chapter_candidate())
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=3,
    )

    response = await service.generate(_request())

    assert len(generator.calls) == 1
    assert generator.calls[0][2] == 3
    assert generator.calls[0][0].child_input == "방구 소리"
    assert len(response.pages) == 2
    assert [len(page.sentences) for page in response.pages] == [4, 4]
    assert response.pages[0].question is None
    assert response.pages[0].requires_branch_input is False
    assert response.pages[-1].question == "표지판은 어느 길을 가리킬까요?"
    assert response.pages[-1].requires_branch_input is True
    assert response.generation.api_call_count == 1
    assert response.generation.page_count == 2
    assert [character.character_id for character in response.pages[0].visual_scene.characters] == [
        "hare",
        "tortoise",
    ]
    assert all(
        character.emotion is not None
        for character in response.pages[0].visual_scene.characters
        if character.present
    )
    hare_scene = next(
        character
        for character in response.pages[0].visual_scene.characters
        if character.character_id == "hare"
    )
    assert hare_scene.emotion is not None
    assert hare_scene.emotion.type == "SURPRISED"
    assert response.state_patch.expected_base_revision == 8
    assert response.state_patch.resolved_facts_added == [
        "아이의 소리가 숲길에 실제로 울려요.",
        "두 친구가 경주로 돌아가요.",
    ]


async def test_question_and_choices_are_analyzed_with_the_final_page() -> None:
    generator = RecordingChapterGenerator(chapter_candidate())
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    response = await service.generate(_request())

    final_quality = response.quality.pages[-1].quality
    assert final_quality.analysis_status == "FULL"
    assert response.quality.chapter.written_syllable_count == sum(
        row.quality.written_syllable_count for row in response.quality.pages
    )
    assert response.quality.chapter.direct_dialogue_count == 2


async def test_forward_question_is_selected_over_question_that_rewinds_story() -> None:
    source = chapter_candidate()
    bad = ChapterCandidate(
        candidate_id="candidate-a-retroactive",
        sentences=source.sentences,
        child_detour_end_sentence_index=4,
        question="출발 종 대신 무엇이 울릴까요?",
        choices=("나무 구호", "잎사귀 종", "조약돌 종"),
    )
    good = ChapterCandidate(
        candidate_id="candidate-z-forward",
        sentences=source.sentences,
        child_detour_end_sentence_index=4,
        question="앞서 간 토끼는 어디에서 쉴까요?",
        choices=(
            "나무 아래서 쉬어요.",
            "바위 곁에서 쉬어요.",
            "시냇물 곁에서 쉬어요.",
        ),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((bad, good)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=2,
    )

    response = await service.generate(_request())

    assert response.generation.selected_candidate_id == good.candidate_id
    assert response.pages[-1].question == good.question
    assert "QUESTION_TIME_REVERSAL" not in (response.quality.pages[-1].quality.contract_failures)
    assert response.generation.api_call_count == 1


async def test_retroactive_question_is_visible_as_best_effort_without_failure() -> None:
    source = chapter_candidate()
    retroactive = ChapterCandidate(
        candidate_id="retroactive-only",
        sentences=source.sentences,
        child_detour_end_sentence_index=4,
        question="출발 종 대신 무엇이 울릴까요?",
        choices=("나무 구호", "잎사귀 종", "조약돌 종"),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((retroactive,)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert response.pages[-1].question == retroactive.question
    assert "QUESTION_TIME_REVERSAL" in (response.quality.pages[-1].quality.contract_failures)
    assert response.quality.chapter.status == "BEST_EFFORT"
    assert response.generation.api_call_count == 1


async def test_quality_contract_retries_until_a_passing_chapter() -> None:
    source = chapter_candidate()
    rejected = ChapterCandidate(
        candidate_id="rejected",
        sentences=source.sentences,
        child_detour_end_sentence_index=4,
        question="출발 종 대신 무엇이 울릴까요?",
        choices=("나무 구호", "잎사귀 종", "조약돌 종"),
    )
    accepted = source
    generator = SequencedChapterGenerator((rejected, accepted))
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
        quality_retry_count=2,
        require_contract_pass=True,
        provider_name="gms",
    )

    response = await service.generate(_request())

    assert generator.calls == 2
    assert response.quality.chapter.contract_pass is True
    assert response.generation.provider == "gms"
    assert response.generation.api_call_count == 2


async def test_quality_contract_blocks_after_two_retries() -> None:
    source = chapter_candidate()
    rejected = ChapterCandidate(
        candidate_id="rejected",
        sentences=source.sentences,
        child_detour_end_sentence_index=4,
        question="출발 종 대신 무엇이 울릴까요?",
        choices=("나무 구호", "잎사귀 종", "조약돌 종"),
    )
    generator = SequencedChapterGenerator((rejected,))
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
        quality_retry_count=2,
        require_contract_pass=True,
        provider_name="gms",
    )

    with pytest.raises(StoryChapterUseCaseError) as raised:
        await service.generate(_request())

    assert generator.calls == 3
    assert raised.value.status_code == 502
    assert raised.value.code == "STORY_QUALITY_CONTRACT_FAILED"


async def test_student_identifier_is_not_forwarded_to_the_model_context() -> None:
    generator = RecordingChapterGenerator(chapter_candidate())
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    await service.generate(_request())

    context_document: dict[str, Any] = generator.calls[0][0].to_dict()
    assert "student_id" not in context_document
    assert "studentId" not in context_document


async def test_v2_dialogue_format_is_part_of_chapter_quality() -> None:
    source = chapter_candidate()
    sentences = list(source.sentences)
    sentences[2] = '"누구 소리야?" 하고 웃어요.'
    generator = RecordingChapterGenerator(
        ChapterCandidate(
            candidate_id="invalid-dialogue",
            sentences=tuple(sentences),
            child_detour_end_sentence_index=4,
            question=source.question,
            choices=source.choices,
        )
    )
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert "CURLY_DIALOGUE_FORMAT" in (response.quality.pages[0].quality.contract_failures)


async def test_malformed_dialogue_uses_one_conditional_repair_call() -> None:
    source = chapter_candidate()
    sentences = list(source.sentences)
    sentences[5] = '거북이는 "난 멈추지 않고 갈 거야!"라고 말해요.'
    sentences[6] = "토끼는 뒤를 보며 거북이를 기다려요."
    repairer = DialogueRepairer()
    generator = RecordingChapterGenerator(
        ChapterCandidate(
            candidate_id="missing-dialogue",
            sentences=tuple(sentences),
            child_detour_end_sentence_index=4,
            question=source.question,
            choices=source.choices,
        )
    )
    service = PersonalizedStoryChapterService(
        generator=generator,
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        repairer=repairer,
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert repairer.calls == 1
    assert response.generation.api_call_count == 2
    assert response.generation.repair_attempted is True
    assert response.generation.repair_accepted is True
    assert response.generation.changed_sentences[0].page_number == 2
    assert response.generation.changed_sentences[0].sentence_number == 2
    assert response.quality.pages[1].quality.contract_pass is True
    tortoise_scene = next(
        character
        for character in response.pages[1].visual_scene.characters
        if character.character_id == "tortoise"
    )
    assert tortoise_scene.action == ("거북이가 “난 멈추지 않고 갈 거야!” 하고 말해요.")


async def test_visual_scene_llm_receives_final_repaired_pages() -> None:
    source = chapter_candidate()
    sentences = list(source.sentences)
    sentences[5] = '거북이는 "난 멈추지 않고 갈 거야!"라고 말해요.'
    sentences[6] = "토끼는 뒤를 보며 거북이를 기다려요."
    planner = RecordingVisualScenePlanner()
    service = PersonalizedStoryChapterService(
        generator=RecordingChapterGenerator(
            ChapterCandidate(
                candidate_id="scene-after-repair",
                sentences=tuple(sentences),
                child_detour_end_sentence_index=4,
                question=source.question,
                choices=source.choices,
            )
        ),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        repairer=DialogueRepairer(),
        visual_scene_planner=planner,
        candidate_count=1,
    )

    response = await service.generate(_request())

    planned_sentences = [sentence for page in planner.pages for sentence in page.sentences]
    assert "거북이가 “난 멈추지 않고 갈 거야!” 하고 말해요." in (planned_sentences)
    assert response.generation.api_call_count == 3
    assert response.generation.visual_scene_status == "LLM_GENERATED"
    assert response.generation.visual_scene_model == "scene-model"
    assert response.timing_ms.visual_scene == 75.0


async def test_visual_scene_failure_returns_transparent_deterministic_fallback() -> None:
    service = PersonalizedStoryChapterService(
        generator=RecordingChapterGenerator(chapter_candidate()),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        visual_scene_planner=FailingVisualScenePlanner(),
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert response.generation.api_call_count == 2
    assert response.generation.visual_scene_status == "DETERMINISTIC_FALLBACK"
    assert response.generation.visual_scene_model == "deterministic"
    assert response.generation.visual_scene_fallback_reason == "TIMEOUT"
    assert [character.character_id for character in response.pages[0].visual_scene.characters] == [
        "hare",
        "tortoise",
    ]


async def test_three_sentence_dynamic_page_can_be_repaired() -> None:
    repairer = RecordingDynamicPageRepairer()
    candidate = ChapterCandidate(
        candidate_id="dynamic-three-sentence-page",
        sentences=(
            "풀밭에 방구 소리가 크게 울려요.",
            "“깜짝 놀랐어!” 토끼가 말해요.",
            "거북이는 웃으며 앞을 바라봐요.",
            "두 친구는 다시 경주 길로 가요.",
            "토끼는 큰 나무 아래에서 잠깐 쉬어요.",
            '거북이는 "멈추지 않고 언덕을 갈 거야!"라고 말해요.',
            "바람은 거북이의 등을 살며시 밀어 줘요.",
            "거북이는 언덕 꼭대기를 향해 계속 걸어요.",
            "“이제 거의 다 왔어.” 거북이가 말해요.",
            "멀리 결승선 깃발이 바람에 흔들려요.",
        ),
        child_detour_end_sentence_index=4,
        question="거북이는 결승선 앞에서 무엇을 할까요?",
        choices=("마지막 힘을 내요", "토끼를 기다려요", "잠깐 숨을 골라요"),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((candidate,)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        repairer=repairer,
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert [len(page.sentences) for page in response.pages] == [4, 3, 3]
    assert len(repairer.calls) == 1
    context, profile, source_candidate = repairer.calls[0]
    assert len(source_candidate.sentences) == 3
    assert context.expected_sentence_count == 3
    assert profile.content_contract.sentence_count == 3
    assert response.generation.repair_attempted is True
    assert response.generation.repair_accepted is True
    assert response.generation.changed_sentences[0].page_number == 2
    assert response.generation.changed_sentences[0].sentence_number == 2


async def test_child_answer_owner_and_later_callback_affect_selection() -> None:
    payload = request_payload(min_pages=2, max_pages=2)
    payload["branchInput"]["text"] = "으쌰으쌰"
    payload["storyState"]["lastQuestion"] = "거북이가 어떤 리듬 말을 외칠까요?"
    request = StoryChapterGenerateRequest.model_validate(payload)
    bad = ChapterCandidate(
        candidate_id="candidate-a-discarded-answer",
        sentences=(
            "으쌰으쌰 소리가 풀밭을 흔들어요.",
            "“그 박자 참 좋다!” 거북이가 말해요.",
            "토끼의 귀가 잠깐 들썩여요.",
            "거북이는 큰 나무 쪽으로 가요.",
            "토끼는 나무 아래에 누워요.",
            "“한숨 자야지.” 토끼가 말해요.",
            "거북이는 토끼 곁을 지나가요.",
            "언덕 위로 긴 길이 이어져요.",
        ),
        child_detour_end_sentence_index=4,
        question="잠에서 깬 토끼가 먼저 뭐라고 외칠까요?",
        choices=("벌써 저기야!", "금방 따라갈 거야!", "이제 달려야 해!"),
    )
    good = ChapterCandidate(
        candidate_id="candidate-z-causal-answer",
        sentences=(
            "거북이는 숨을 모아 발을 내디뎌요.",
            "“으쌰으쌰!” 거북이가 걸음마다 외쳐요.",
            "그 박자에 풀잎이 좌우로 흔들려요.",
            "토끼의 귀도 박자에 맞춰 들썩여요.",
            "그 소리를 듣던 토끼는 나무 아래에 누워요.",
            "“잠깐만 쉬어야지.” 토끼가 말해요.",
            "거북이는 같은 걸음으로 토끼 곁을 지나가요.",
            "언덕 위로 긴 길이 이어져요.",
        ),
        child_detour_end_sentence_index=4,
        question="잠에서 깬 토끼가 먼저 뭐라고 외칠까요?",
        choices=("벌써 저기야!", "금방 따라갈 거야!", "이제 달려야 해!"),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((bad, good)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=2,
    )

    response = await service.generate(request)

    assert response.generation.selected_candidate_id == good.candidate_id


def test_catalog_branch_placeholders_are_materialized_before_prompting() -> None:
    payload = request_payload(min_pages=2, max_pages=2)
    payload["branchInput"]["text"] = "으쌰으쌰"
    payload["storyTemplate"]["currentBeat"]["goal"] = (
        "직전 답의 리듬을 이어 가며 거북이가 토끼를 지나가요."
    )
    payload["chapterPlan"]["orderedEvents"][0]["lockedEvent"] = (
        "거북이가 직전 답의 리듬 말을 되뇌며 다가가요."
    )
    payload["chapterPlan"]["orderedEvents"][0]["requiredConcepts"] = ["직전 답의 리듬 말"]
    request = StoryChapterGenerateRequest.model_validate(payload)

    context = build_chapter_generation_context(request)

    serialized = " ".join((context.chapter_goal, *context.ordered_events))
    assert "으쌰으쌰" in serialized
    assert "직전 답" not in serialized


async def test_missing_child_input_is_visible_as_contract_failure() -> None:
    source = chapter_candidate()
    missing = ChapterCandidate(
        candidate_id="missing-child-input",
        sentences=tuple(
            sentence.replace("방구 소리", "나뭇잎 소리") for sentence in source.sentences
        ),
        child_detour_end_sentence_index=4,
        question=source.question,
        choices=source.choices,
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((missing,)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    response = await service.generate(_request())

    assert "CHILD_INPUT_NOT_REFLECTED" in (response.quality.pages[0].quality.contract_failures)


async def test_question_answer_already_in_body_is_contract_failure() -> None:
    payload = request_payload(min_pages=2, max_pages=2)
    payload.pop("branchInput")
    request = StoryChapterGenerateRequest.model_validate(payload)
    candidate = ChapterCandidate(
        candidate_id="question-answer-leak",
        sentences=(
            "거북이는 작은 숨에 맞춰 숫자를 세어요.",
            "“하나, 둘, 하나, 둘.” 거북이가 말해요.",
            "토끼는 앞서 멈칫하다가 다시 웃어요.",
            "거북이는 토끼 곁을 지나 계속 가요.",
            "거북이는 종을 바라봐요.",
            "“조금만 더 가자.” 거북이가 말해요.",
            "거북이는 조용히 길을 걸어요.",
            "언덕 끝에 갈림길이 나타나요.",
        ),
        question="거북이가 언덕에서 무엇을 먼저 할까요?",
        choices=("종을 바라봐요.", "조용히 걸어요.", "친구를 불러요."),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((candidate,)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=1,
    )

    response = await service.generate(request)

    assert "QUESTION_ANSWER_ALREADY_IN_BODY" in (
        response.quality.pages[-1].quality.contract_failures
    )


async def test_rhythm_question_prefers_body_that_keeps_words_undecided() -> None:
    payload = request_payload(min_pages=2, max_pages=2)
    payload.pop("branchInput")
    request = StoryChapterGenerateRequest.model_validate(payload)
    shared_tail = (
        "토끼는 나무 아래에 누워 쉬어요.",
        "“곧 따라잡을 거야.” 토끼가 말해요.",
        "거북이는 잠든 토끼 곁에 다가가요.",
        "거북이는 다음 걸음을 준비해요.",
    )
    bad = ChapterCandidate(
        candidate_id="candidate-a-rhythm-consumed",
        sentences=(
            "거북이는 작은 숨에 맞춰 숫자를 세어요.",
            "“하나, 둘, 하나, 둘.” 거북이가 말해요.",
            "토끼는 앞서 달리며 크게 웃어요.",
            "거북이는 제 걸음으로 언덕을 올라요.",
            *shared_tail,
        ),
        question="거북이가 어떤 리듬 말을 흥얼거릴까요?",
        choices=("힘차게 가자!", "끝까지 가자!", "천천히 가자!"),
    )
    good = ChapterCandidate(
        candidate_id="candidate-z-rhythm-open",
        sentences=(
            "거북이는 걸음마다 짧게 숨을 쉬어요.",
            "“내 걸음에 맞는 말이 필요해.” 거북이가 말해요.",
            "토끼는 앞서 달리며 크게 웃어요.",
            "거북이는 첫 리듬 말을 고르려 해요.",
            *shared_tail,
        ),
        question="거북이가 어떤 리듬 말을 흥얼거릴까요?",
        choices=("힘차게 가자!", "끝까지 가자!", "천천히 가자!"),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((bad, good)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=2,
    )

    response = await service.generate(request)

    assert response.generation.selected_candidate_id == good.candidate_id


async def test_dialogue_focus_rejects_question_that_switches_to_action() -> None:
    payload = request_payload(min_pages=2, max_pages=2)
    payload.pop("branchInput")
    focus = "잠에서 깬 토끼가 처음 외칠 짧은 말의 내용"
    payload["storyTemplate"]["currentBeat"]["questionFocus"] = focus
    payload["chapterPlan"]["questionFocus"] = focus
    request = StoryChapterGenerateRequest.model_validate(payload)
    source = chapter_candidate(child_input=False)
    bad = ChapterCandidate(
        candidate_id="candidate-a-action-question",
        sentences=source.sentences,
        question="잠에서 깬 토끼가 가장 먼저 무엇을 할까요?",
        choices=(
            "거북이를 따라가요.",
            "그늘에 더 누워요.",
            "물을 마시러 가요.",
        ),
    )
    good = ChapterCandidate(
        candidate_id="candidate-z-dialogue-question",
        sentences=source.sentences,
        question="잠에서 깬 토끼가 뭐라고 외칠까요?",
        choices=("벌써 앞섰어?", "잠깐만 기다려!", "어디까지 갔지?"),
    )
    service = PersonalizedStoryChapterService(
        generator=FixedChapterBatchGenerator((bad, good)),
        analyzer=DeterministicAnalyzer(),  # type: ignore[arg-type]
        candidate_count=2,
    )

    response = await service.generate(request)

    assert response.generation.selected_candidate_id == good.candidate_id
