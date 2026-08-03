from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from iread_ai.application.chapter_comparison_builder import (
    build_displayed_chapter_comparison_payload,
)
from iread_ai.config import Settings
from iread_ai.contracts.story_chapter import StoryChapterGenerateRequest
from iread_ai.devtools.dynamic_story_simulator import (
    build_chapter_request,
    initial_dynamic_runtime,
)
from iread_ai.devtools.service_story_catalog import (
    STORY_CATALOG,
    ServiceStoryFixture,
    StoryBeatFixture,
)

PROFILE_KEYS = ("balanced", "beginner")
STAGE_LABELS = ("opening", "early", "middle", "late", "ending")
DEFAULT_OUTPUT_ROOT = (
    Path("local-output") / "chapter-comparison-batch"
)
DEFAULT_GENERATE_ENDPOINT = (
    "http://127.0.0.1:8081/api/v3/story/chapters/generate"
)
DEFAULT_COMPARISON_ENDPOINT = (
    "http://127.0.0.1:8081/api/dev/story/"
    "displayed-chapter-comparison"
)
SCORE_EPSILON = 0.005

_SLOT_ANSWERS: dict[str, str] = {
    "ACTION_ORDER": "작은 것부터 하나씩 해 보자.",
    "APOLOGY": "미안해. 내가 다시 고칠게.",
    "BASKET_ITEM": "향기 좋은 작은 꽃을 넣을래.",
    "BOUNDARY_PHRASE": "장난은 멈추고 함께 이야기하자.",
    "BUILDING_TOOL": "튼튼한 망치를 빌려줄래.",
    "CHEER_PHRASE": "힘내, 천천히 끝까지 가자!",
    "COMFORT_ITEM": "따뜻한 풀잎 이불을 챙길래.",
    "DECORATION": "세 집에 같은 별 깃발을 달자.",
    "DESCRIPTION": "손잡이가 낡은 작은 쇠도끼예요.",
    "DIALOGUE": "우리 함께 해 보자!",
    "FAIR_RULE": "한 번씩 차례를 지키자.",
    "FAREWELL": "멀리 있어도 우리는 좋은 친구야.",
    "FOOD_DETAIL": "달콤한 열매 한입을 먹을래.",
    "GIFT_USE": "이웃의 부러진 울타리를 고쳐 줄래.",
    "GREETING": "안녕, 만나서 정말 반가워!",
    "HELPFUL_ITEM": "큰 잎을 우산처럼 쓸래.",
    "HELP_METHOD": "친구와 힘을 모아 도울래.",
    "HELP_REQUEST": "친구야, 내 소리를 들으면 와 줘.",
    "HONEST_ANSWER": "그건 제 도끼가 아니에요.",
    "HOUSE_USE": "작은 무대를 먼저 만들래.",
    "INSTRUMENT": "통통 울리는 작은 북을 칠래.",
    "MEMORY": "엄마의 다정한 말을 기억할래.",
    "MELODY": "높고 맑은 소리를 맡을래.",
    "MOVEMENT_STYLE": "두 발을 천천히 저으며 균형을 잡을래.",
    "MUSIC_ROLE": "신나는 박자를 맡아 줘.",
    "OBSERVATION": "발자국이 할머니 집 쪽으로 이어져요.",
    "PATH_MARKER": "반짝이는 돌을 길 표시로 놓을래.",
    "PERFORMANCE_ORDER": "당나귀, 개, 고양이, 수탉 차례로 연주하자.",
    "PROMISE": "작아도 꼭 도와줄게!",
    "REPAIR_ACTION": "무너진 곳에 새 벽돌을 나를래.",
    "REST_PLACE": "따뜻한 나무 그늘에서 쉬고 싶어요.",
    "ROPE_TARGET": "가장 느슨한 매듭부터 갉을래.",
    "SAFE_ACTION": "먼저 멈추고 친구에게 도움을 청할래.",
    "SAFE_CHECK": "창밖에서 조용히 살펴볼래.",
    "SAFE_DIALOGUE": "나는 약속한 길로 갈 거야.",
    "SAFE_PLACE": "식탁 아래 작은 상자 뒤에 숨을래.",
    "SAFE_ROUTE": "큰 잎을 타고 낮은 길로 갈래.",
    "SAFETY_CHECK": "우리만 아는 암호를 물어볼래.",
    "SEARCH_METHOD": "긴 나뭇가지에 자석을 달아 찾을래.",
    "SECRET_SIGNAL": "파란 꽃 이름을 암호로 정할래.",
    "SELF_TALK": "하나 둘, 천천히 가자.",
    "SENSORY_DETAIL": "포근한 풀 냄새가 나는 곳이 좋아요.",
    "SIGNAL": "도와줘, 짹짹! 하고 외칠래.",
    "SIGNAL_SOUND": "톡톡 후두둑, 비가 와!",
    "SONG_STYLE": "하나 둘, 함께 가자!",
    "START_SIGNAL": "이제 네 차례야, 반짝!",
    "STOP_SIGNAL": "구름 종이 울리면 멈추자.",
    "STRENGTH_DETAIL": "큰 발로 물을 힘차게 밀 수 있어.",
    "TEAM_ACTION": "모두 같은 박자로 힘을 모으자.",
    "TEAM_NAME": "반짝 숲 음악대라고 부를래.",
    "TEAM_POSITION": "창문 아래에 차례로 서 볼래.",
    "THANKS": "네가 와 줘서 정말 고마워!",
    "TRAVEL_ITEM": "작은 손수건을 챙길래.",
    "WALKING_RHYTHM": "하나 둘, 천천히 끝까지!",
    "WEATHER_STYLE": "따뜻한 햇살과 살랑바람을 만들자.",
    "WELCOME_PHRASE": "어서 와, 네가 와서 기뻐!",
}


def normalized_chapter_indexes(
    story: ServiceStoryFixture,
) -> tuple[int, int, int, int, int]:
    last = story.total_chapters - 1
    indexes = tuple(round(last * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1))
    if len(set(indexes)) != 5:
        raise ValueError(
            f"story {story.template_id} cannot provide five unique stages"
        )
    return indexes


def _answer_for_beat(beat: StoryBeatFixture) -> str:
    for slot in beat.allowed_branch_slots:
        if slot in _SLOT_ANSWERS:
            return _SLOT_ANSWERS[slot]
    return "우리 함께 천천히 해 보자!"


def _input_for_stage(
    *,
    story: ServiceStoryFixture,
    chapter_index: int,
    stage_index: int,
) -> tuple[dict[str, str] | None, str]:
    if chapter_index == 0:
        return None, "NONE"
    previous_beat = story.beats[chapter_index - 1]
    answer = _answer_for_beat(previous_beat)
    if stage_index in {1, 4}:
        return {"source": "CHOICE", "text": answer}, "CHOICE_ON_TOPIC"
    if stage_index == 2:
        return (
            {"source": "TEXT_CONFIRMED", "text": answer},
            "FREE_TEXT_ON_TOPIC",
        )
    return (
        {
            "source": "TEXT_CONFIRMED",
            "text": f"방귀 소리가 뿡 나요. 그래도 {answer}",
        },
        "FREE_TEXT_PLAYFUL",
    )


def _synthetic_runtime(
    story: ServiceStoryFixture,
    *,
    profile_key: str,
    chapter_index: int,
    student_id: int,
) -> dict[str, Any]:
    runtime = initial_dynamic_runtime(
        story,
        profile_key=profile_key,
        student_id=student_id,
    )
    runtime["storyRevision"] = chapter_index
    runtime["lastAppliedChapterNumber"] = chapter_index
    if chapter_index == 0:
        return runtime

    prior_beats = story.beats[:chapter_index]
    previous_beat = prior_beats[-1]
    last_question = previous_beat.question_focus
    if not last_question:
        raise ValueError(
            f"story {story.template_id} chapter {chapter_index + 1} "
            "has no previous branch question"
        )
    state = runtime["storyState"]
    state["rollingSummary"] = " ".join(
        beat.goal for beat in prior_beats
    )
    state["resolvedFacts"] = [
        beat.goal for beat in prior_beats
    ]
    state["unresolvedHooks"] = [last_question]
    state["lastQuestion"] = last_question
    state["recentPages"] = [
        {
            "pageNumber": page_number,
            "sentences": [page.locked_event],
            "question": (
                last_question
                if page_number == len(previous_beat.pages)
                else None
            ),
        }
        for page_number, page in enumerate(
            previous_beat.pages,
            start=1,
        )
    ]
    return runtime


def build_cases(
    *,
    run_token: str = "fixture",
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    safe_token = "".join(
        character
        for character in run_token
        if character.isalnum() or character in {"-", "_"}
    )[:24] or "fixture"
    for story_index, story in enumerate(STORY_CATALOG):
        for stage_index, chapter_index in enumerate(
            normalized_chapter_indexes(story)
        ):
            branch_input, input_type = _input_for_stage(
                story=story,
                chapter_index=chapter_index,
                stage_index=stage_index,
            )
            student_id = 10_000 + story_index * 10 + stage_index
            for profile_key in PROFILE_KEYS:
                case_id = (
                    f"{story.template_id}-{profile_key}-"
                    f"{STAGE_LABELS[stage_index]}"
                )
                request_id = (
                    f"cb-{safe_token}-{story.template_id}-"
                    f"{profile_key}-{stage_index + 1}"
                )
                runtime = _synthetic_runtime(
                    story,
                    profile_key=profile_key,
                    chapter_index=chapter_index,
                    student_id=student_id,
                )
                request = build_chapter_request(
                    story,
                    runtime,
                    chapter_index,
                    branch_input,
                    request_id,
                )
                StoryChapterGenerateRequest.model_validate(request)
                cases.append(
                    {
                        "caseId": case_id,
                        "storyId": story.template_id,
                        "storyTitle": story.title,
                        "profileKey": profile_key,
                        "stage": STAGE_LABELS[stage_index],
                        "stageIndex": stage_index,
                        "chapterNumber": chapter_index + 1,
                        "totalChapters": story.total_chapters,
                        "inputType": input_type,
                        "childInput": (
                            branch_input["text"]
                            if branch_input is not None
                            else None
                        ),
                        "idempotencyKey": request_id,
                        "chapterRequest": request,
                    }
                )
    if len(cases) != 100:
        raise RuntimeError(
            f"chapter comparison matrix must contain 100 cases, got {len(cases)}"
        )
    return cases


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _error_document(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


async def _post_json(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    response = await client.post(
        endpoint,
        headers=headers,
        json=dict(payload),
    )
    if response.status_code >= 400:
        try:
            detail: Any = response.json()
        except ValueError:
            detail = response.text[:2000]
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{json.dumps(detail, ensure_ascii=False)}"
        )
    document = response.json()
    if not isinstance(document, dict):
        raise ValueError("API response must be a JSON object")
    return document


async def _run_case(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: dict[str, Any],
    generate_endpoint: str,
    comparison_endpoint: str,
    api_key: str,
    raw_dir: Path,
    existing: dict[str, Any] | None,
    deadline: float,
) -> dict[str, Any]:
    raw_path = raw_dir / f"{case['caseId']}.json"
    record = existing or {
        "case": case,
        "status": "PENDING",
        "personalized": None,
        "comparison": None,
        "error": None,
    }
    if record.get("status") == "SUCCESS":
        return record
    async with semaphore:
        if time.monotonic() >= deadline:
            record["status"] = "SKIPPED_DEADLINE"
            record["error"] = {
                "type": "DeadlineExceeded",
                "message": "batch deadline reached",
            }
            _atomic_json(raw_path, record)
            return record
        personalized_stage = record.get("personalized")
        if not isinstance(personalized_stage, dict) or not isinstance(
            personalized_stage.get("response"),
            dict,
        ):
            started = time.perf_counter()
            try:
                response = await _post_json(
                    client,
                    endpoint=generate_endpoint,
                    api_key=api_key,
                    payload=case["chapterRequest"],
                    idempotency_key=str(case["idempotencyKey"]),
                )
            except Exception as exc:
                record["status"] = "PERSONALIZED_ERROR"
                record["error"] = _error_document(exc)
                _atomic_json(raw_path, record)
                return record
            record["personalized"] = {
                "response": response,
                "elapsedMs": round(
                    (time.perf_counter() - started) * 1000,
                    3,
                ),
            }
            record["status"] = "PERSONALIZED_READY"
            record["error"] = None
            _atomic_json(raw_path, record)

        personalized_response = record["personalized"]["response"]
        comparison_payload = build_displayed_chapter_comparison_payload(
            case["chapterRequest"],
            personalized_response,
            request_id=f"cmp-{case['chapterRequest']['requestId']}",
        )
        started = time.perf_counter()
        try:
            comparison_response = await _post_json(
                client,
                endpoint=comparison_endpoint,
                api_key=api_key,
                payload=comparison_payload,
            )
        except Exception as exc:
            record["status"] = "COMPARISON_ERROR"
            record["error"] = _error_document(exc)
            _atomic_json(raw_path, record)
            return record
        record["comparison"] = {
            "response": comparison_response,
            "elapsedMs": round(
                (time.perf_counter() - started) * 1000,
                3,
            ),
        }
        record["status"] = "SUCCESS"
        record["error"] = None
        _atomic_json(raw_path, record)
        return record


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int = 10_000,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        rounded = round(float(values[0]), 4)
        return [rounded, rounded]
    randomizer = random.Random(seed)
    sample_size = len(values)
    means = [
        statistics.fmean(
            values[randomizer.randrange(sample_size)]
            for _ in range(sample_size)
        )
        for _ in range(iterations)
    ]
    return [
        round(_percentile(means, 0.025) or 0.0, 4),
        round(_percentile(means, 0.975) or 0.0, 4),
    ]


def _exact_two_sided_binomial(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    tail = min(wins, losses)
    probability = (
        2.0
        * sum(math.comb(trials, count) for count in range(tail + 1))
        / (2**trials)
    )
    return round(min(1.0, probability), 8)


def numeric_statistics(
    values: Sequence[float],
    *,
    seed: int,
) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if not finite:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "bootstrapMean95": None,
        }
    return {
        "n": len(finite),
        "min": round(min(finite), 4),
        "max": round(max(finite), 4),
        "mean": round(statistics.fmean(finite), 4),
        "median": round(statistics.median(finite), 4),
        "p05": round(_percentile(finite, 0.05) or 0.0, 4),
        "p95": round(_percentile(finite, 0.95) or 0.0, 4),
        "bootstrapMean95": _bootstrap_mean_interval(
            finite,
            seed=seed,
        ),
    }


def _flatten_page_text(outcome: Mapping[str, Any]) -> str:
    pages = outcome.get("pages")
    if not isinstance(pages, list):
        return ""
    lines: list[str] = []
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        sentences = page.get("sentences")
        if isinstance(sentences, list):
            lines.extend(
                str(sentence)
                for sentence in sentences
                if isinstance(sentence, str)
            )
    return "\n".join(lines)


def _fit(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    value = outcome.get("fit")
    return value if isinstance(value, Mapping) else {}


def _timing(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    value = outcome.get("timingMs")
    return value if isinstance(value, Mapping) else {}


def _generation(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    value = outcome.get("generation")
    return value if isinstance(value, Mapping) else {}


def _comparison_row(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if record.get("status") != "SUCCESS":
        return None
    case = record.get("case")
    comparison_stage = record.get("comparison")
    if not isinstance(case, Mapping) or not isinstance(
        comparison_stage,
        Mapping,
    ):
        return None
    response = comparison_stage.get("response")
    if not isinstance(response, Mapping):
        return None
    plain = response.get("plain")
    personalized = response.get("personalized")
    comparison = response.get("comparison")
    diagnostics = response.get("diagnostics")
    if not all(
        isinstance(value, Mapping)
        for value in (plain, personalized, comparison, diagnostics)
    ):
        return None
    plain = dict(plain)
    personalized = dict(personalized)
    comparison = dict(comparison)
    diagnostics = dict(diagnostics)
    delta = comparison.get("delta")
    if not isinstance(delta, Mapping):
        return None
    plain_fit = _fit(plain)
    personalized_fit = _fit(personalized)
    plain_generation = _generation(plain)
    personalized_generation = _generation(personalized)
    plain_timing = _timing(plain)
    personalized_timing = _timing(personalized)
    score_delta = _number(delta.get("profileFitScore"))
    if score_delta is None:
        return None
    score_basis = str(comparison.get("scoreBasis", "UNKNOWN"))
    risk_key = (
        "surfaceRiskPer10"
        if score_basis == "COMMON_SURFACE_ONLY"
        else "riskPer10"
    )
    plain_written_syllables = _number(
        plain_fit.get("writtenSyllables")
    )
    personalized_written_syllables = _number(
        personalized_fit.get("writtenSyllables")
    )
    return {
        "caseId": str(case["caseId"]),
        "storyId": int(case["storyId"]),
        "storyTitle": str(case["storyTitle"]),
        "profileKey": str(case["profileKey"]),
        "stage": str(case["stage"]),
        "chapterNumber": int(case["chapterNumber"]),
        "inputType": str(case["inputType"]),
        "childInput": case.get("childInput"),
        "scoreBasis": score_basis,
        "comparisonConfidence": str(
            comparison.get("comparisonConfidence", "UNKNOWN")
        ),
        "winner": str(comparison.get("winner", "UNVERIFIED")),
        "plainProfileFitScore": _number(
            comparison.get("plainProfileFitScore")
        ),
        "personalizedProfileFitScore": _number(
            comparison.get("personalizedProfileFitScore")
        ),
        "profileFitScoreDelta": score_delta,
        "plainComparedRiskPer10": _number(plain_fit.get(risk_key)),
        "personalizedComparedRiskPer10": _number(
            personalized_fit.get(risk_key)
        ),
        "riskPer10Delta": _number(delta.get("riskPer10")),
        "plainWrittenSyllables": plain_written_syllables,
        "personalizedWrittenSyllables": (
            personalized_written_syllables
        ),
        "writtenSyllableDelta": (
            personalized_written_syllables
            - plain_written_syllables
            if (
                personalized_written_syllables is not None
                and plain_written_syllables is not None
            )
            else None
        ),
        "plainExcludedOverage": _number(
            plain_fit.get("excludedOverage")
        ),
        "personalizedExcludedOverage": _number(
            personalized_fit.get("excludedOverage")
        ),
        "excludedOverageDelta": _number(delta.get("excludedOverage")),
        "plainLimitedOverage": _number(
            plain_fit.get("limitedOverage")
        ),
        "personalizedLimitedOverage": _number(
            personalized_fit.get("limitedOverage")
        ),
        "limitedOverageDelta": _number(delta.get("limitedOverage")),
        "targetDistanceDelta": _number(delta.get("targetDistance")),
        "latencyDeltaMs": _number(delta.get("totalElapsedMs")),
        "plainContractPass": bool(plain_fit.get("contractPass")),
        "plainContractFailures": list(
            plain_fit.get("contractFailures", [])
        ),
        "personalizedContractPass": bool(
            personalized_fit.get("contractPass")
        ),
        "personalizedContractFailures": list(
            personalized_fit.get("contractFailures", [])
        ),
        "plainAnalysisStatus": plain_fit.get("analysisStatus"),
        "personalizedAnalysisStatus": personalized_fit.get(
            "analysisStatus"
        ),
        "plainUnverifiedSkills": plain_fit.get(
            "unverifiedSkillCodes",
            [],
        ),
        "personalizedUnverifiedSkills": personalized_fit.get(
            "unverifiedSkillCodes",
            [],
        ),
        "plainText": _flatten_page_text(plain),
        "personalizedText": _flatten_page_text(personalized),
        "plainTotalMs": _number(plain_timing.get("total")),
        "personalizedTotalMs": _number(
            personalized_timing.get("total")
        ),
        "plainModelCalls": int(
            _number(plain_generation.get("apiCallCount")) or 0
        ),
        "personalizedModelCalls": int(
            _number(personalized_generation.get("apiCallCount")) or 0
        ),
        "personalizedCandidateCount": int(
            _number(personalized_generation.get("candidateCount")) or 0
        ),
        "personalizedRepairAttempted": bool(
            personalized_generation.get("repairAttempted")
        ),
        "personalizedRepairAccepted": bool(
            personalized_generation.get("repairAccepted")
        ),
        "newComparisonModelCalls": int(
            _number(diagnostics.get("newApiCallCount")) or 0
        ),
    }


def _case_excerpt(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "caseId",
            "storyTitle",
            "profileKey",
            "stage",
            "chapterNumber",
            "inputType",
            "childInput",
            "scoreBasis",
            "comparisonConfidence",
            "plainProfileFitScore",
            "personalizedProfileFitScore",
            "profileFitScoreDelta",
            "riskPer10Delta",
            "plainUnverifiedSkills",
            "personalizedUnverifiedSkills",
            "plainText",
            "personalizedText",
        )
    }


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    metrics = (
        "profileFitScoreDelta",
        "riskPer10Delta",
        "writtenSyllableDelta",
        "excludedOverageDelta",
        "limitedOverageDelta",
        "targetDistanceDelta",
        "latencyDeltaMs",
    )
    ordered = sorted(
        rows,
        key=lambda row: float(row["profileFitScoreDelta"]),
    )
    wins = sum(
        float(row["profileFitScoreDelta"]) > SCORE_EPSILON
        for row in rows
    )
    losses = sum(
        float(row["profileFitScoreDelta"]) < -SCORE_EPSILON
        for row in rows
    )
    ties = len(rows) - wins - losses
    return {
        "n": len(rows),
        "scoreBasisCounts": dict(
            Counter(str(row["scoreBasis"]) for row in rows)
        ),
        "confidenceCounts": dict(
            Counter(
                str(row["comparisonConfidence"])
                for row in rows
            )
        ),
        "winTieLoss": {
            "personalizedWins": wins,
            "ties": ties,
            "plainWins": losses,
            "personalizedWinRate": (
                round(wins / len(rows), 4) if rows else 0.0
            ),
            "exactSignP": _exact_two_sided_binomial(wins, losses),
        },
        "plainContractPassRate": (
            round(
                sum(bool(row["plainContractPass"]) for row in rows)
                / len(rows),
                4,
            )
            if rows
            else 0.0
        ),
        "personalizedContractPassRate": (
            round(
                sum(
                    bool(row["personalizedContractPass"])
                    for row in rows
                )
                / len(rows),
                4,
            )
            if rows
            else 0.0
        ),
        "plainContractFailureCounts": dict(
            Counter(
                str(failure)
                for row in rows
                for failure in row["plainContractFailures"]
            )
        ),
        "personalizedContractFailureCounts": dict(
            Counter(
                str(failure)
                for row in rows
                for failure in row[
                    "personalizedContractFailures"
                ]
            )
        ),
        "repair": {
            "attempted": sum(
                bool(row["personalizedRepairAttempted"])
                for row in rows
            ),
            "accepted": sum(
                bool(row["personalizedRepairAccepted"])
                for row in rows
            ),
            "attemptRate": (
                round(
                    sum(
                        bool(row["personalizedRepairAttempted"])
                        for row in rows
                    )
                    / len(rows),
                    4,
                )
                if rows
                else 0.0
            ),
            "acceptRateAmongAttempts": (
                round(
                    sum(
                        bool(row["personalizedRepairAccepted"])
                        for row in rows
                    )
                    / sum(
                        bool(row["personalizedRepairAttempted"])
                        for row in rows
                    ),
                    4,
                )
                if any(
                    bool(row["personalizedRepairAttempted"])
                    for row in rows
                )
                else 0.0
            ),
        },
        "conditionStatistics": {
            metric: numeric_statistics(
                [
                    float(row[metric])
                    for row in rows
                    if row.get(metric) is not None
                ],
                seed=seed + sum(ord(character) for character in metric),
            )
            for metric in (
                "plainProfileFitScore",
                "personalizedProfileFitScore",
                "plainComparedRiskPer10",
                "personalizedComparedRiskPer10",
                "plainWrittenSyllables",
                "personalizedWrittenSyllables",
                "plainExcludedOverage",
                "personalizedExcludedOverage",
                "plainLimitedOverage",
                "personalizedLimitedOverage",
                "plainTotalMs",
                "personalizedTotalMs",
            )
        },
        "metrics": {
            metric: numeric_statistics(
                [
                    float(row[metric])
                    for row in rows
                    if row.get(metric) is not None
                ],
                seed=seed + sum(ord(character) for character in metric),
            )
            for metric in metrics
        },
        "highestCase": (
            _case_excerpt(ordered[-1]) if ordered else None
        ),
        "lowestCase": (
            _case_excerpt(ordered[0]) if ordered else None
        ),
        "topFive": [
            _case_excerpt(row)
            for row in reversed(ordered[-5:])
        ],
        "bottomFive": [
            _case_excerpt(row)
            for row in ordered[:5]
        ],
        "topFiveMeanDelta": (
            round(
                statistics.fmean(
                    float(row["profileFitScoreDelta"])
                    for row in ordered[-5:]
                ),
                4,
            )
            if ordered
            else None
        ),
        "bottomFiveMeanDelta": (
            round(
                statistics.fmean(
                    float(row["profileFitScoreDelta"])
                    for row in ordered[:5]
                ),
                4,
            )
            if ordered
            else None
        ),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        row
        for record in records
        if (row := _comparison_row(record)) is not None
    ]

    def grouped(field: str) -> dict[str, Any]:
        keys = sorted({str(row[field]) for row in rows})
        return {
            key: _aggregate_rows(
                [
                    row
                    for row in rows
                    if str(row[field]) == key
                ],
                seed=seed + sum(ord(character) for character in key),
            )
            for key in keys
        }

    model_calls = sum(
        int(row["personalizedModelCalls"])
        + int(row["newComparisonModelCalls"])
        for row in rows
    )
    successful = sum(record.get("status") == "SUCCESS" for record in records)
    overall = _aggregate_rows(rows, seed=seed)
    rows_by_basis = {
        basis: [
            row for row in rows if str(row["scoreBasis"]) == basis
        ]
        for basis in ("FULL_POLICY", "COMMON_SURFACE_ONLY")
    }
    by_score_basis = {
        basis: _aggregate_rows(
            basis_rows,
            seed=seed + sum(ord(character) for character in basis),
        )
        for basis, basis_rows in rows_by_basis.items()
    }
    summary = {
        "schemaVersion": 1,
        "experiment": "paired-v3-chapter-personalization-100-v1",
        "generatedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "requestedCaseCount": len(records),
        "successfulCaseCount": successful,
        "comparableCaseCount": len(rows),
        "failedCaseCount": len(records) - successful,
        "comparisonUnavailableCount": successful - len(rows),
        "successRate": (
            round(successful / len(records), 4) if records else 0.0
        ),
        "elapsedSeconds": round(elapsed_seconds, 3),
        "imageApiCallCount": 0,
        "visualSceneJsonIncluded": True,
        "modelCallCount": model_calls,
        "reliability": {
            "fullPolicyCount": len(rows_by_basis["FULL_POLICY"]),
            "commonSurfaceOnlyCount": len(
                rows_by_basis["COMMON_SURFACE_ONLY"]
            ),
            "unavailableCount": successful - len(rows),
        },
        "primaryEffect": {
            "basis": "FULL_POLICY",
            "n": by_score_basis["FULL_POLICY"]["n"],
            "riskPer10Delta": by_score_basis["FULL_POLICY"][
                "metrics"
            ]["riskPer10Delta"],
            "profileFitScoreDelta": by_score_basis["FULL_POLICY"][
                "metrics"
            ]["profileFitScoreDelta"],
            "winTieLoss": by_score_basis["FULL_POLICY"]["winTieLoss"],
            "highestCase": by_score_basis["FULL_POLICY"][
                "highestCase"
            ],
            "lowestCase": by_score_basis["FULL_POLICY"][
                "lowestCase"
            ],
        },
        "partialSurfaceEffect": {
            "basis": "COMMON_SURFACE_ONLY",
            "n": by_score_basis["COMMON_SURFACE_ONLY"]["n"],
            "riskPer10Delta": by_score_basis[
                "COMMON_SURFACE_ONLY"
            ]["metrics"]["riskPer10Delta"],
            "profileFitScoreDelta": by_score_basis[
                "COMMON_SURFACE_ONLY"
            ]["metrics"]["profileFitScoreDelta"],
            "winTieLoss": by_score_basis[
                "COMMON_SURFACE_ONLY"
            ]["winTieLoss"],
            "highestCase": by_score_basis[
                "COMMON_SURFACE_ONLY"
            ]["highestCase"],
            "lowestCase": by_score_basis[
                "COMMON_SURFACE_ONLY"
            ]["lowestCase"],
        },
        "overall": overall,
        "byProfile": grouped("profileKey"),
        "byStory": grouped("storyTitle"),
        "byStage": grouped("stage"),
        "byInputType": grouped("inputType"),
        "byScoreBasis": by_score_basis,
        "errors": [
            {
                "caseId": record.get("case", {}).get("caseId"),
                "status": record.get("status"),
                "error": record.get("error"),
            }
            for record in records
            if record.get("status") != "SUCCESS"
        ],
        "interpretationCaveat": (
            "Kiwi·G2P 기반 점수는 후보 선택에도 사용됩니다. 이 실험은 "
            "동일한 자동 평가 기준에서의 개인화 향상을 측정하며, 독립적인 "
            "교육 효과를 증명하지 않습니다. FULL_POLICY와 "
            "COMMON_SURFACE_ONLY 결과를 분리해 해석해야 합니다."
        ),
    }
    return summary, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, list | dict)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    overall = summary["overall"]
    score = overall["metrics"]["profileFitScoreDelta"]
    condition = overall["conditionStatistics"]
    full = summary["primaryEffect"]
    partial = summary["partialSurfaceEffect"]
    lines = [
        "# v3 장 생성 개인화 100건 비교",
        "",
        (
            f"- 성공: {summary['successfulCaseCount']}/"
            f"{summary['requestedCaseCount']}"
        ),
        f"- 비교 가능: {summary['comparableCaseCount']}건",
        f"- 모델 호출: {summary['modelCallCount']}회",
        "- 이미지 API 호출: 0회",
        "- visualScene JSON 생성 호출: 포함",
        "",
        "## 신뢰도별 개인화 효과",
        "",
        (
            "| 점수 근거 | n | 적합도 저점 | 적합도 평균 | 적합도 중앙 "
            "| 적합도 고점 | 부담/10 평균 변화 | 개인화 승/무/패 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| FULL_POLICY | "
            f"{full['n']} | "
            f"{full['profileFitScoreDelta']['min']} | "
            f"{full['profileFitScoreDelta']['mean']} | "
            f"{full['profileFitScoreDelta']['median']} | "
            f"{full['profileFitScoreDelta']['max']} | "
            f"{full['riskPer10Delta']['mean']} | "
            f"{full['winTieLoss']['personalizedWins']}/"
            f"{full['winTieLoss']['ties']}/"
            f"{full['winTieLoss']['plainWins']} |"
        ),
        (
            "| COMMON_SURFACE_ONLY | "
            f"{partial['n']} | "
            f"{partial['profileFitScoreDelta']['min']} | "
            f"{partial['profileFitScoreDelta']['mean']} | "
            f"{partial['profileFitScoreDelta']['median']} | "
            f"{partial['profileFitScoreDelta']['max']} | "
            f"{partial['riskPer10Delta']['mean']} | "
            f"{partial['winTieLoss']['personalizedWins']}/"
            f"{partial['winTieLoss']['ties']}/"
            f"{partial['winTieLoss']['plainWins']} |"
        ),
        "",
        "## 전체 참고값",
        "",
        (
            "- 서로 다른 점수 근거를 합친 참고값: "
            f"평균 {score['mean']}, 중앙 {score['median']}, "
            f"최저 {score['min']}, 최고 {score['max']}, "
            f"P05 {score['p05']}, P95 {score['p95']}"
        ),
        (
            "- 승/무/패: "
            f"{overall['winTieLoss']['personalizedWins']}/"
            f"{overall['winTieLoss']['ties']}/"
            f"{overall['winTieLoss']['plainWins']}"
        ),
        (
            "- 평균 글 길이: 일반 "
            f"{condition['plainWrittenSyllables']['mean']}음절 → 개인화 "
            f"{condition['personalizedWrittenSyllables']['mean']}음절"
        ),
        (
            "- 평균 생성 시간: 일반 "
            f"{condition['plainTotalMs']['mean']}ms → 개인화 "
            f"{condition['personalizedTotalMs']['mean']}ms"
        ),
        (
            "- 계약 통과율: 일반 "
            f"{overall['plainContractPassRate'] * 100:.1f}% → 개인화 "
            f"{overall['personalizedContractPassRate'] * 100:.1f}%"
        ),
        (
            "- 국소 교정 시도/채택: "
            f"{overall['repair']['attempted']}/"
            f"{overall['repair']['accepted']}"
        ),
        (
            "- 점수 근거: `"
            + json.dumps(
                overall["scoreBasisCounts"],
                ensure_ascii=False,
            )
            + "`"
        ),
        "",
        "## 최고 사례",
        "",
        "```json",
        json.dumps(
            overall["highestCase"],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 최저 사례",
        "",
        "```json",
        json.dumps(
            overall["lowestCase"],
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## 해석 주의",
        "",
        str(summary["interpretationCaveat"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_existing_records(
    raw_dir: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        record = _read_json(path)
        case = record.get("case")
        if not isinstance(case, Mapping) or not case.get("caseId"):
            raise ValueError(f"raw record has no caseId: {path}")
        records[str(case["caseId"])] = record
    return records


def _parse_args() -> argparse.Namespace:
    configured_api_key = os.getenv("INTERNAL_API_KEY")
    if not configured_api_key:
        configured_api_key = (
            Settings().internal_api_key.get_secret_value()
        )
    parser = argparse.ArgumentParser(
        description=(
            "Run 100 paired current-v3 chapter personalization comparisons."
        )
    )
    parser.add_argument(
        "--generate-endpoint",
        default=DEFAULT_GENERATE_ENDPOINT,
    )
    parser.add_argument(
        "--comparison-endpoint",
        default=DEFAULT_COMPARISON_ENDPOINT,
    )
    parser.add_argument(
        "--api-key",
        default=configured_api_key,
    )
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-minutes", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument(
        "--confirm-real-api",
        action="store_true",
        help="Required acknowledgement that this run makes paid model calls.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.confirm_real_api:
        raise ValueError(
            "real API execution requires --confirm-real-api"
        )
    if not 1 <= args.case_limit <= 100:
        raise ValueError("case-limit must be between 1 and 100")
    if not 1 <= args.concurrency <= 2:
        raise ValueError("concurrency must be between 1 and 2")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if args.max_minutes <= 0:
        raise ValueError("max-minutes must be positive")


def _validate_real_provider_health(health: object) -> str:
    provider = (
        health.get("storyProvider")
        if isinstance(health, Mapping)
        else None
    )
    if (
        not isinstance(health, Mapping)
        or health.get("status") not in {"UP", "ok"}
        or provider not in {"gms", "openai"}
    ):
        raise RuntimeError(
            "preflight requires a healthy API with "
            "storyProvider=gms or openai"
        )
    return str(provider)


async def _run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    health_endpoint = (
        args.generate_endpoint.partition("/api/")[0].rstrip("/")
        + "/health"
    )
    async with httpx.AsyncClient(timeout=10.0) as health_client:
        health_response = await health_client.get(health_endpoint)
        health_response.raise_for_status()
        health = health_response.json()
    story_provider = _validate_real_provider_health(health)

    if args.resume_dir is not None:
        run_dir = args.resume_dir.resolve()
        cases_document = json.loads(
            (run_dir / "cases.json").read_text(encoding="utf-8")
        )
        if not isinstance(cases_document, list):
            raise ValueError("cases.json must contain an array")
        cases = [
            dict(case)
            for case in cases_document
            if isinstance(case, Mapping)
        ]
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_token = datetime.now().strftime("%Y%m%d-%H%M%S")
        cases = build_cases(run_token=run_token)
        random.Random(args.seed).shuffle(cases)
        cases = cases[: args.case_limit]
        run_dir = (
            args.output_root.resolve()
            / f"{run_token}_{len(cases)}cases"
        )
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(run_dir / "cases.json", cases)

    existing = _load_existing_records(raw_dir)
    pending = [
        case
        for case in cases
        if existing.get(str(case["caseId"]), {}).get("status")
        != "SUCCESS"
    ]
    metadata = {
        "schemaVersion": 1,
        "generateEndpoint": args.generate_endpoint,
        "comparisonEndpoint": args.comparison_endpoint,
        "storyProvider": story_provider,
        "caseCount": len(cases),
        "pendingCaseCount": len(pending),
        "concurrency": args.concurrency,
        "timeoutSeconds": args.timeout_seconds,
        "maxMinutes": args.max_minutes,
        "seed": args.seed,
        "resumed": args.resume_dir is not None,
        "startedAt": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }
    _atomic_json(run_dir / "run.json", metadata)

    started = time.perf_counter()
    deadline = time.monotonic() + args.max_minutes * 60
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(
        max_connections=max(4, args.concurrency * 2),
        max_keepalive_connections=max(2, args.concurrency),
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:
        tasks = [
            asyncio.create_task(
                _run_case(
                    client=client,
                    semaphore=semaphore,
                    case=case,
                    generate_endpoint=args.generate_endpoint,
                    comparison_endpoint=args.comparison_endpoint,
                    api_key=args.api_key,
                    raw_dir=raw_dir,
                    existing=existing.get(str(case["caseId"])),
                    deadline=deadline,
                )
            )
            for case in pending
        ]
        for completed, task in enumerate(
            asyncio.as_completed(tasks),
            start=1,
        ):
            record = await task
            print(
                f"[{completed:03d}/{len(pending):03d}] "
                f"{record['case']['caseId']} · {record['status']}",
                flush=True,
            )

    records_by_id = _load_existing_records(raw_dir)
    records = [
        records_by_id.get(
            str(case["caseId"]),
            {
                "case": case,
                "status": "MISSING",
                "error": {
                    "type": "MissingRecord",
                    "message": "case produced no raw record",
                },
            },
        )
        for case in cases
    ]
    elapsed_seconds = time.perf_counter() - started
    summary, rows = aggregate_records(
        records,
        seed=args.seed,
        elapsed_seconds=elapsed_seconds,
    )
    _atomic_json(run_dir / "summary.json", summary)
    _write_csv(run_dir / "case_rows.csv", rows)
    _write_markdown(run_dir / "SUMMARY.md", summary)
    metadata.update(
        {
            "finishedAt": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "elapsedSeconds": round(elapsed_seconds, 3),
            "successfulCaseCount": summary["successfulCaseCount"],
            "comparableCaseCount": summary["comparableCaseCount"],
        }
    )
    _atomic_json(run_dir / "run.json", metadata)
    return run_dir


def main() -> None:
    args = _parse_args()
    run_dir = asyncio.run(_run(args))
    print(f"RESULT_DIR={run_dir}", flush=True)


if __name__ == "__main__":
    main()
