from __future__ import annotations

import os
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
import streamlit as st

from iread_ai.devtools.curriculum_samples import (
    CURRICULUM_SAMPLE_PROFILES,
    curriculum_sample,
)
from iread_ai.devtools.training_playground import (
    BUILD_TYPES,
    CHOICE_TYPES,
    SERVICE_TEMPLATE_IDS,
    TRACE_TYPES,
    build_candidate_request,
    build_target_features,
    candidate_texts,
    choice_label,
    correct_choice,
    expected_text,
    prompt_text,
    service_training_groups,
    service_training_spec_by_id,
)

DEFAULT_API_BASE = os.getenv("IREAD_TRAINING_API_BASE_URL", "http://127.0.0.1:8081")
DEFAULT_API_KEY = os.getenv("AI_INTERNAL_API_KEY", "iread-local-ai-key")


class PlaygroundAPIError(RuntimeError):
    pass


@st.cache_resource
def api_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(75.0, connect=5.0))


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "play_curriculum": None,
        "play_curriculum_meta": None,
        "play_materials": [],
        "play_answers": {},
        "play_profile_name": next(iter(CURRICULUM_SAMPLE_PROFILES)),
        "play_run_id": uuid.uuid4().hex[:10],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_run() -> None:
    st.session_state.play_curriculum = None
    st.session_state.play_curriculum_meta = None
    st.session_state.play_materials = []
    st.session_state.play_answers = {}
    st.session_state.play_run_id = uuid.uuid4().hex[:10]


def post_json(
    api_base: str,
    api_key: str,
    path: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str], float]:
    endpoint = f"{api_base.rstrip('/')}{path}"
    request_id = str(payload["requestId"])
    started = time.perf_counter()
    try:
        response = api_client().post(
            endpoint,
            headers={
                "X-API-Key": api_key.strip(),
                "Idempotency-Key": request_id,
                "Content-Type": "application/json",
            },
            json=dict(payload),
        )
    except httpx.RequestError as exception:
        raise PlaygroundAPIError(f"AI 서버에 연결하지 못했습니다: {exception}") from exception
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code >= 400:
        try:
            error = response.json()
        except ValueError:
            error = response.text
        raise PlaygroundAPIError(f"HTTP {response.status_code}: {error}")
    return (
        response.json(),
        {
            "provider": response.headers.get("X-AI-Provider", "unknown"),
            "fallback": response.headers.get("X-AI-Fallback", ""),
            "replayed": response.headers.get("Idempotent-Replayed", "false"),
        },
        elapsed_ms,
    )


def generate_curriculum(
    api_base: str,
    api_key: str,
    profile_name: str,
    use_llm: bool,
) -> None:
    sample = curriculum_sample(profile_name)
    request_id = f"play-curriculum-{st.session_state.play_run_id}-{uuid.uuid4().hex[:6]}"
    payload = {
        "requestId": request_id,
        "schemaVersion": 1,
        "featureProfiles": sample["featureProfiles"],
        "recentTrainings": sample["recentTrainings"],
        "useLlm": use_llm,
    }
    result, headers, elapsed_ms = post_json(
        api_base,
        api_key,
        "/api/v1/curricula/recommend",
        payload,
    )
    st.session_state.play_curriculum = result
    st.session_state.play_curriculum_meta = {
        **headers,
        "elapsedMs": elapsed_ms,
        "request": payload,
    }
    st.session_state.play_materials = []
    st.session_state.play_answers = {}


def material_request(
    recommendation: Mapping[str, Any],
    sample: Mapping[str, Any],
    sequence: int,
    use_lexicon: bool,
) -> dict[str, Any]:
    target_features = build_target_features(
        sample.get("featureProfiles", []),
        recommendation.get("targetFeatureCodes", []),
    )
    request_id = (
        f"play-material-{st.session_state.play_run_id}-{sequence}-{uuid.uuid4().hex[:6]}"
    )
    return build_candidate_request(
        request_id=request_id,
        training_type=str(recommendation["trainingType"]),
        difficulty=int(recommendation.get("recommendedDifficulty", 2)),
        target_features=target_features,
        additional_prompt=(
            "같은 회차의 다른 훈련과 구별되는 내용으로 만들고, "
            "각 후보는 서로 다른 소재와 낱말을 사용하세요."
        ),
        use_lexicon=use_lexicon,
    )


def generate_all_materials(
    api_base: str,
    api_key: str,
    profile_name: str,
    use_lexicon: bool,
) -> None:
    curriculum = st.session_state.play_curriculum or {}
    recommendations = curriculum.get("recommendations", [])
    sample = curriculum_sample(profile_name)
    generated: list[dict[str, Any]] = []
    with st.status("추천 교안 5개를 생성하고 있습니다.", expanded=True) as status:
        for sequence, recommendation in enumerate(recommendations, start=1):
            template_id = int(recommendation["trainingTemplateId"])
            if template_id not in SERVICE_TEMPLATE_IDS:
                status.write(
                    f"{sequence}. {recommendation['trainingName']}: 서비스 미등록 템플릿이라 건너뜀"
                )
                continue
            status.write(f"{sequence}. {recommendation['trainingName']} 생성 중")
            payload = material_request(recommendation, sample, sequence, use_lexicon)
            try:
                response, headers, elapsed_ms = post_json(
                    api_base,
                    api_key,
                    "/api/v1/trainings/candidates",
                    payload,
                )
            except PlaygroundAPIError as exception:
                generated.append(
                    {
                        "recommendation": dict(recommendation),
                        "request": payload,
                        "error": str(exception),
                    }
                )
                status.write(f"{sequence}. 생성 실패: {exception}")
                continue
            generated.append(
                {
                    "recommendation": dict(recommendation),
                    "request": payload,
                    "response": response,
                    "headers": headers,
                    "elapsedMs": elapsed_ms,
                }
            )
            status.write(
                f"{sequence}. 완료 · {headers['provider']} · {elapsed_ms / 1000:.2f}초"
            )
        failures = sum("error" in item for item in generated)
        if failures:
            status.update(
                label=f"교안 생성 완료 · 성공 {len(generated) - failures}개, 실패 {failures}개",
                state="error",
            )
        else:
            status.update(label="추천 교안 5개 생성 완료", state="complete")
    st.session_state.play_materials = generated
    st.session_state.play_answers = {}


def generate_manual_material(
    api_base: str,
    api_key: str,
    profile_name: str,
    template_id: int,
    difficulty: int,
    use_lexicon: bool,
) -> None:
    spec = service_training_spec_by_id(template_id)
    if spec is None:
        raise PlaygroundAPIError("서비스에 등록된 훈련을 선택해 주세요.")
    sample = curriculum_sample(profile_name)
    suggested_codes = [spec.suggested_feature]
    recommendation = {
        "sequenceNo": 1,
        "trainingTemplateId": spec.template_id,
        "trainingType": spec.training_type,
        "trainingName": spec.name,
        "role": "MANUAL",
        "recommendedDifficulty": difficulty,
        "targetFeatureCodes": suggested_codes,
        "rationale": "전체 훈련 목록에서 직접 선택한 테스트입니다.",
    }
    payload = build_candidate_request(
        request_id=f"play-manual-{st.session_state.play_run_id}-{uuid.uuid4().hex[:6]}",
        training_type=spec.training_type,
        difficulty=difficulty,
        target_features=build_target_features(
            sample.get("featureProfiles", []),
            suggested_codes,
        ),
        additional_prompt="후보 5개는 서로 다른 소재와 낱말을 사용하세요.",
        use_lexicon=use_lexicon,
    )
    response, headers, elapsed_ms = post_json(
        api_base,
        api_key,
        "/api/v1/trainings/candidates",
        payload,
    )
    st.session_state.play_materials = [
        {
            "recommendation": recommendation,
            "request": payload,
            "response": response,
            "headers": headers,
            "elapsedMs": elapsed_ms,
        }
    ]
    st.session_state.play_answers = {}


def answer_key(material_index: int, candidate_index: int) -> str:
    return f"{material_index}:{candidate_index}"


def save_answer(material_index: int, candidate_index: int, correct: bool) -> None:
    st.session_state.play_answers[answer_key(material_index, candidate_index)] = correct


def render_choice(
    training_type: str,
    candidate: Mapping[str, Any],
    widget_key: str,
    material_index: int,
    candidate_index: int,
) -> None:
    choices = candidate.get("choices", [])
    labels = [choice_label(choice) for choice in choices]
    selected = st.segmented_control(
        "정답을 골라 보세요",
        labels,
        key=f"choice-{widget_key}",
        width="stretch",
    )
    if st.button(
        "정답 확인",
        key=f"check-{widget_key}",
        type="primary",
        icon=":material/check:",
    ):
        expected = correct_choice(candidate)
        correct = selected is not None and selected == expected
        save_answer(material_index, candidate_index, correct)
    stored = st.session_state.play_answers.get(answer_key(material_index, candidate_index))
    if stored is True:
        st.success("정답입니다.")
    elif stored is False:
        st.error(f"다시 확인해 보세요. 정답: {correct_choice(candidate)}")
    if training_type == "FILL_IN_THE_BLANK":
        st.caption(f"완성 문장: {candidate.get('completedSentence', '')}")


def render_build(
    training_type: str,
    candidate: Mapping[str, Any],
    widget_key: str,
    material_index: int,
    candidate_index: int,
) -> None:
    selected_indices: list[int] = []
    for label, field in (
        ("첫소리", "initialChoices"),
        ("가운데소리", "medialChoices"),
        ("끝소리", "finalChoices"),
    ):
        values = candidate.get(field)
        if not isinstance(values, list):
            continue
        selected = st.segmented_control(
            label,
            list(range(len(values))),
            format_func=lambda index, values=values: str(values[index]),
            key=f"{field}-{widget_key}",
        )
        selected_indices.append(-1 if selected is None else int(selected))
    if st.button(
        "글자 완성",
        key=f"build-{widget_key}",
        type="primary",
        icon=":material/extension:",
    ):
        expected_indices = [
            int(candidate[field])
            for field in ("initialAnswerIndex", "medialAnswerIndex", "finalAnswerIndex")
            if field in candidate
        ]
        save_answer(material_index, candidate_index, selected_indices == expected_indices)
    stored = st.session_state.play_answers.get(answer_key(material_index, candidate_index))
    if stored is True:
        st.success(f"완성했어요: {expected_text(training_type, candidate)}")
    elif stored is False:
        st.error(f"다시 조합해 보세요. 완성 글자: {expected_text(training_type, candidate)}")


def render_text_answer(
    training_type: str,
    candidate: Mapping[str, Any],
    widget_key: str,
    material_index: int,
    candidate_index: int,
) -> None:
    if isinstance(candidate.get("cards"), list):
        st.pills(
            "사용할 카드",
            [str(value) for value in candidate["cards"]],
            selection_mode="multi",
            key=f"cards-{widget_key}",
            disabled=True,
        )
    if isinstance(candidate.get("syllables"), list):
        st.write(" · ".join(str(value) for value in candidate["syllables"]))
    response = st.text_input(
        "완성한 글자 또는 문장",
        key=f"text-{widget_key}",
        placeholder="정답을 입력하세요",
    )
    if st.button(
        "정답 확인",
        key=f"text-check-{widget_key}",
        type="primary",
        icon=":material/check:",
    ):
        expected = expected_text(training_type, candidate)
        def normalized(value: object) -> str:
            return "".join(str(value).split()).rstrip(".?!")

        save_answer(
            material_index,
            candidate_index,
            normalized(response) == normalized(expected),
        )
    stored = st.session_state.play_answers.get(answer_key(material_index, candidate_index))
    if stored is True:
        st.success("정답입니다.")
    elif stored is False:
        st.error(f"정답: {expected_text(training_type, candidate)}")


def render_reading(
    training_type: str,
    candidate: Mapping[str, Any],
    widget_key: str,
    material_index: int,
    candidate_index: int,
) -> None:
    if candidate.get("title"):
        st.subheader(str(candidate["title"]))
    values = candidate_texts(candidate)
    if training_type == "PHRASE_READING" and candidate.get("phrases"):
        st.info(" / ".join(str(value) for value in candidate["phrases"]))
    else:
        for value in values:
            st.markdown(f"### {value}")
    if training_type == "REPEATED_SENTENCE_READING":
        st.caption(f"같은 문장을 {candidate.get('repeatCount', 2)}번 읽습니다.")
    recording = st.audio_input(
        "읽은 목소리 녹음",
        key=f"audio-{widget_key}",
        sample_rate=16000,
    )
    if st.button(
        "읽기 완료",
        key=f"read-{widget_key}",
        type="primary",
        icon=":material/mic:",
    ):
        save_answer(material_index, candidate_index, recording is not None)
    stored = st.session_state.play_answers.get(answer_key(material_index, candidate_index))
    if stored is True:
        st.success(
            "녹음이 준비되었습니다. 이 앱에서는 실제 발음 평가 API를 "
            "자동 호출하지 않습니다."
        )
    elif stored is False:
        st.warning("먼저 목소리를 녹음해 주세요.")


def render_trace(
    candidate: Mapping[str, Any],
    widget_key: str,
    material_index: int,
    candidate_index: int,
) -> None:
    st.markdown(f"# {candidate.get('target', '')}")
    st.caption(f"읽는 소리: {candidate.get('soundText', '')}")
    traced = st.checkbox("눈으로 획을 따라갔어요", key=f"trace-{widget_key}")
    recording = st.audio_input(
        "글자 소리 녹음",
        key=f"trace-audio-{widget_key}",
        sample_rate=16000,
    )
    if st.button(
        "따라 보기 완료",
        key=f"trace-done-{widget_key}",
        type="primary",
        icon=":material/draw:",
    ):
        save_answer(material_index, candidate_index, traced and recording is not None)
    stored = st.session_state.play_answers.get(answer_key(material_index, candidate_index))
    if stored is True:
        st.success("따라 보기와 녹음을 완료했습니다.")
    elif stored is False:
        st.warning("따라 보기 확인과 녹음이 모두 필요합니다.")


def render_candidate(
    material: Mapping[str, Any],
    material_index: int,
    candidate_index: int,
) -> None:
    recommendation = material["recommendation"]
    training_type = str(recommendation["trainingType"])
    candidate = material["response"]["data"][candidate_index]
    widget_key = f"{material_index}-{candidate_index}-{training_type}"
    with st.container(border=True):
        st.caption(f"문항 {candidate_index + 1}/5 · {training_type}")
        st.subheader(prompt_text(training_type, candidate))
        if candidate.get("audioText"):
            st.info(f"소리 안내: {candidate['audioText']}", icon=":material/volume_up:")
        if training_type in TRACE_TYPES:
            render_trace(candidate, widget_key, material_index, candidate_index)
        elif training_type in CHOICE_TYPES:
            render_choice(
                training_type,
                candidate,
                widget_key,
                material_index,
                candidate_index,
            )
        elif training_type in BUILD_TYPES:
            render_build(
                training_type,
                candidate,
                widget_key,
                material_index,
                candidate_index,
            )
        elif training_type in {
            "SYLLABLE_BLEND",
            "FINAL_CONSONANT_DELETE",
            "SYLLABLE_DELETE",
            "SYLLABLE_REPLACE",
            "SENTENCE_ASSEMBLY",
        }:
            render_text_answer(
                training_type,
                candidate,
                widget_key,
                material_index,
                candidate_index,
            )
        else:
            render_reading(
                training_type,
                candidate,
                widget_key,
                material_index,
                candidate_index,
            )


def material_rows(materials: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for material in materials:
        recommendation = material.get("recommendation", {})
        response = material.get("response", {})
        metadata = response.get("generationMetadata") or {}
        rows.append(
            {
                "순서": recommendation.get("sequenceNo"),
                "훈련": recommendation.get("trainingName"),
                "유형": recommendation.get("trainingType"),
                "역할": recommendation.get("role"),
                "provider": material.get("headers", {}).get("provider", "실패"),
                "전략": metadata.get("strategy", "-"),
                "시간(초)": round(float(material.get("elapsedMs", 0)) / 1000, 2),
                "문항": len(response.get("data", [])),
                "오류": material.get("error", ""),
            }
        )
    return rows


st.set_page_config(
    page_title="iRead 훈련 통합 테스트",
    page_icon=":material/school:",
    layout="wide",
)
initialize_state()

st.title("iRead 훈련 통합 테스트")
st.caption(
    "실제 AI 커리큘럼 추천과 교안 생성 API를 호출하고, "
    "운영 DB의 31개 훈련 형식을 직접 풀어보는 개발 도구입니다."
)

with st.sidebar:
    st.header("API 설정")
    api_base = st.text_input("AI API 주소", value=DEFAULT_API_BASE, key="play-api-base")
    api_key = st.text_input(
        "X-API-Key",
        value=DEFAULT_API_KEY,
        type="password",
        key="play-api-key",
    )
    profile_name = st.selectbox(
        "아동 프로필",
        list(CURRICULUM_SAMPLE_PROFILES),
        key="play_profile_name",
    )
    use_llm = st.toggle("커리큘럼에 LLM 사용", value=True)
    use_lexicon = st.toggle("교안에 어휘 DB 사용", value=True)
    if st.button("새 테스트 시작", icon=":material/restart_alt:"):
        reset_run()
        st.rerun()
    st.caption("이미지 생성 API는 호출하지 않습니다.")

sample = curriculum_sample(profile_name)
with st.container(border=True):
    st.subheader(profile_name)
    st.write(CURRICULUM_SAMPLE_PROFILES[profile_name]["description"])
    if sample["featureProfiles"]:
        st.dataframe(
            [
                {
                    "특징": value["featureCode"],
                    "정확도": value.get("accuracyRate"),
                    "취약도": value.get("weaknessScore"),
                    "신뢰도": value.get("confidence"),
                    "근거 수": value.get("evidenceCount"),
                }
                for value in sample["featureProfiles"]
            ],
            hide_index=True,
        )
    else:
        st.info("아직 평가 근거가 없는 신규 학습자 프로필입니다.")

curriculum_tab, material_tab, play_tab = st.tabs(
    ["1. 커리큘럼 추천", "2. 교안 생성", "3. 훈련 체험"]
)

with curriculum_tab:
    st.subheader("다음 회차 훈련 5개 추천")
    st.write("선택한 아동 프로필로 실제 커리큘럼 추천 API를 호출합니다.")
    if st.button(
        "커리큘럼 생성",
        type="primary",
        icon=":material/auto_awesome:",
        key="generate-curriculum",
    ):
        try:
            with st.skeleton(height=220):
                generate_curriculum(api_base, api_key, profile_name, use_llm)
        except PlaygroundAPIError as exception:
            st.error(str(exception))
        else:
            st.rerun()

    curriculum = st.session_state.play_curriculum
    if curriculum:
        meta = st.session_state.play_curriculum_meta or {}
        metrics = st.columns(4)
        metrics[0].metric("현재 단계", curriculum["currentStage"])
        metrics[1].metric("허용 단계", curriculum["maximumAllowedStage"])
        metrics[2].metric("provider", meta.get("provider", "-"))
        metrics[3].metric("생성 시간", f"{meta.get('elapsedMs', 0) / 1000:.2f}초")
        st.dataframe(
            [
                {
                    "순서": item["sequenceNo"],
                    "역할": item["role"],
                    "훈련": item["trainingName"],
                    "난이도": item["recommendedDifficulty"],
                    "점수": item["score"],
                    "목표 특징": ", ".join(item["targetFeatureCodes"]) or "없음",
                    "추천 이유": item["rationale"],
                }
                for item in curriculum["recommendations"]
            ],
            hide_index=True,
        )
        missing = [
            item
            for item in curriculum["recommendations"]
            if int(item["trainingTemplateId"]) not in SERVICE_TEMPLATE_IDS
        ]
        if missing:
            st.error(
                "운영 DB에 없는 템플릿이 추천됐습니다: "
                + ", ".join(item["trainingName"] for item in missing)
            )
        if curriculum.get("warnings"):
            for warning in curriculum["warnings"]:
                st.warning(warning)
        with st.expander("요청·응답 JSON"):
            st.json(meta.get("request", {}))
            st.json(curriculum)
    else:
        st.info("먼저 커리큘럼을 생성해 주세요.")

with material_tab:
    st.subheader("교안 문항 생성")
    mode = st.segmented_control(
        "생성 방식",
        ["추천 5개 한꺼번에", "전체 목록에서 하나 선택"],
        default="추천 5개 한꺼번에",
        key="material-mode",
    )
    if mode == "추천 5개 한꺼번에":
        if not st.session_state.play_curriculum:
            st.info("1단계에서 커리큘럼을 먼저 생성해 주세요.")
        elif st.button(
            "추천 교안 5개 생성",
            type="primary",
            icon=":material/library_add:",
            key="generate-materials",
        ):
            generate_all_materials(api_base, api_key, profile_name, use_lexicon)
    else:
        groups = service_training_groups()
        selected_group = st.selectbox("훈련 영역", list(groups), key="manual-group")
        specs = groups[selected_group]
        selected_id = st.selectbox(
            "훈련",
            [spec.template_id for spec in specs],
            format_func=lambda value: f"{value}. {service_training_spec_by_id(value).name}",
            key="manual-template",
        )
        difficulty = st.slider("난이도", 1, 5, 2, key="manual-difficulty")
        if st.button(
            "선택 교안 생성",
            type="primary",
            icon=":material/add_task:",
            key="generate-manual",
        ):
            try:
                with st.skeleton(height=180):
                    generate_manual_material(
                        api_base,
                        api_key,
                        profile_name,
                        int(selected_id),
                        difficulty,
                        use_lexicon,
                    )
            except PlaygroundAPIError as exception:
                st.error(str(exception))
            else:
                st.rerun()

    materials = st.session_state.play_materials
    if materials:
        st.dataframe(material_rows(materials), hide_index=True)
        with st.expander("교안 요청·응답 원문"):
            for index, material in enumerate(materials, start=1):
                st.markdown(f"**{index}. {material['recommendation']['trainingName']}**")
                st.json(material)
    else:
        st.info("생성된 교안이 없습니다.")

with play_tab:
    materials = [item for item in st.session_state.play_materials if "response" in item]
    if not materials:
        st.info("2단계에서 교안을 생성하면 여기에서 직접 풀어볼 수 있습니다.")
    else:
        labels = [
            f"{index + 1}. {item['recommendation']['trainingName']}"
            for index, item in enumerate(materials)
        ]
        selected_label = st.selectbox("체험할 훈련", labels, key="play-material-select")
        material_index = labels.index(selected_label)
        material = materials[material_index]
        recommendation = material["recommendation"]
        response = material["response"]
        candidates = response["data"]
        answered = sum(
            answer_key(material_index, index) in st.session_state.play_answers
            for index in range(len(candidates))
        )
        correct = sum(
            st.session_state.play_answers.get(answer_key(material_index, index)) is True
            for index in range(len(candidates))
        )
        metrics = st.columns(4)
        metrics[0].metric("훈련", recommendation["trainingName"])
        metrics[1].metric("완료", f"{answered}/{len(candidates)}")
        metrics[2].metric("정답", correct)
        metrics[3].metric("provider", material["headers"]["provider"])
        st.progress(answered / len(candidates), text="훈련 진행률")
        page_slot = st.container()
        with st.container(horizontal=True, horizontal_alignment="center"):
            page = st.pagination(
                len(candidates),
                key=f"candidate-page-{material_index}",
                persist_state="session",
            )
        with page_slot:
            render_candidate(material, material_index, page - 1)
        with st.expander("현재 문항 진단"):
            st.write(f"목표 특징: {', '.join(recommendation['targetFeatureCodes']) or '없음'}")
            st.write(f"생성 시간: {material['elapsedMs'] / 1000:.2f}초")
            st.json(candidates[page - 1])

st.divider()
st.caption(
    "이 앱의 정답 확인은 화면 동작 검증용입니다. "
    "녹음의 실제 발음·유창성 평가는 별도 평가 API에서 수행합니다."
)
