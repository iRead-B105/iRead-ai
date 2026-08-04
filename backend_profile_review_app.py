from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from iread_ai.application.reading_profile_request_adapter import (
    build_curriculum_recommend_request,
    build_teacher_report_request,
)
from iread_ai.contracts.reading_profile import StudentReadingProfileSnapshot
from iread_ai.devtools.backend_profile_samples import backend_profile_sample

load_dotenv()

st.set_page_config(
    page_title="iRead 백엔드 프로필 연동 검토",
    page_icon=":material/account_child_invert:",
    layout="wide",
)


def _default_api_key() -> str:
    return (
        os.getenv("AI_INTERNAL_API_KEY", "").strip()
        or os.getenv("AI_API_KEY", "").strip()
        or "local-development-key"
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _reset_inputs() -> None:
    sample = backend_profile_sample()
    st.session_state.backend_profile_editor = _pretty_json(sample["featureProfiles"])
    st.session_state.backend_feature_labels_editor = _pretty_json(sample["featureLabels"])
    st.session_state.backend_gaze_trend_editor = _pretty_json(sample["gazeTrend"])
    st.session_state.backend_recent_trainings_editor = _pretty_json(sample["recentTrainings"])
    st.session_state.backend_profile_validation = None
    st.session_state.backend_teacher_result = None
    st.session_state.backend_curriculum_result = None


def _initialize_state() -> None:
    sample = backend_profile_sample()
    defaults = {
        "backend_profile_editor": _pretty_json(sample["featureProfiles"]),
        "backend_feature_labels_editor": _pretty_json(sample["featureLabels"]),
        "backend_gaze_trend_editor": _pretty_json(sample["gazeTrend"]),
        "backend_recent_trainings_editor": _pretty_json(sample["recentTrainings"]),
        "backend_profile_validation": None,
        "backend_teacher_result": None,
        "backend_curriculum_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _parse_json(text: str, *, label: str, expected_type: type) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exception:
        raise ValueError(
            f"{label} JSON 문법 오류: {exception.msg} (줄 {exception.lineno})"
        ) from exception
    if not isinstance(value, expected_type):
        expected = "배열" if expected_type is list else "객체"
        raise ValueError(f"{label}은(는) JSON {expected}여야 합니다.")
    return value


def _post_json(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
        response = client.post(
            endpoint,
            headers={
                "X-API-Key": api_key.strip(),
                "Idempotency-Key": payload["requestId"],
                "Content-Type": "application/json",
            },
            json=payload,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    try:
        body = response.json()
    except ValueError:
        body = {"message": response.text or "응답 본문이 없습니다."}
    if response.is_error:
        message = body.get("message") if isinstance(body, dict) else str(body)
        raise RuntimeError(f"HTTP {response.status_code}: {message}")
    return {
        "request": payload,
        "response": body,
        "elapsedMs": elapsed_ms,
        "provider": response.headers.get("X-AI-Provider", ""),
        "fallback": response.headers.get("X-AI-Fallback", ""),
    }


def _validated_inputs(
    profile_text: str,
    feature_labels_text: str,
    gaze_trend_text: str,
    recent_trainings_text: str,
) -> tuple[StudentReadingProfileSnapshot, dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    profiles = _parse_json(profile_text, label="읽기 프로필", expected_type=list)
    feature_labels = _parse_json(
        feature_labels_text,
        label="특징 이름",
        expected_type=dict,
    )
    gaze_trend = _parse_json(gaze_trend_text, label="시선 추세", expected_type=dict)
    recent_trainings = _parse_json(
        recent_trainings_text,
        label="최근 훈련",
        expected_type=list,
    )
    snapshot = StudentReadingProfileSnapshot.model_validate(
        {"featureProfiles": profiles}
    )
    normalized_labels = {str(key): str(value) for key, value in feature_labels.items()}
    return snapshot, normalized_labels, gaze_trend, recent_trainings


def _render_teacher_result(result: dict[str, Any]) -> None:
    body = result["response"]
    provider = body.get("summaryProvider") or result.get("provider") or "-"
    st.caption(f"요약 방식: `{provider}`")
    metrics = st.columns(3)
    metrics[0].metric("데이터 충분도", body.get("dataSufficiency", "-"))
    metrics[1].metric("향상 특징", len(body.get("improvedPatterns", [])))
    metrics[2].metric("처리 시간", f"{result['elapsedMs']:.1f}ms")

    with st.container(border=True):
        st.subheader("교수자 관찰 요약")
        improved = body.get("improvedPatterns", [])
        persistent = body.get("persistentDifficultyPatterns", [])
        if improved:
            st.markdown("**향상된 패턴**")
            for text in improved:
                st.markdown(f"- {text}")
        if persistent:
            st.markdown("**지속 관찰이 필요한 패턴**")
            for text in persistent:
                st.markdown(f"- {text}")
        if not improved and not persistent:
            st.info("표시할 읽기 특징 판단이 없습니다.")

        gaze = body.get("gazeDescriptions", {})
        if gaze:
            st.markdown("**시선 관찰**")
            for source, descriptions in gaze.items():
                for text in descriptions:
                    st.markdown(f"- `{source}` · {text}")

    with st.expander("교수자 분석 요청·응답 JSON", expanded=False):
        st.caption("요청")
        st.json(result["request"])
        st.caption("응답")
        st.json(body)


def _render_curriculum_result(result: dict[str, Any]) -> None:
    body = result["response"]
    provider = body.get("recommendationProvider") or result.get("provider") or "-"
    st.caption(f"추천 방식: `{provider}`")
    metrics = st.columns(3)
    metrics[0].metric("현재 단계", body.get("currentStage", "-"))
    metrics[1].metric("최대 허용 단계", body.get("maximumAllowedStage", "-"))
    metrics[2].metric("처리 시간", f"{result['elapsedMs']:.1f}ms")

    if result.get("fallback"):
        st.warning(f"Fallback: {result['fallback']}")
    for warning in body.get("warnings", []):
        st.warning(warning)

    recommendations = body.get("recommendations", [])
    rows = [
        {
            "순서": item.get("sequenceNo"),
            "훈련 ID": item.get("trainingTemplateId"),
            "훈련명": item.get("trainingName"),
            "역할": item.get("role"),
            "단계": item.get("curriculumStage"),
            "난이도": item.get("recommendedDifficulty"),
            "점수": item.get("score"),
            "목표 특징": ", ".join(item.get("targetFeatureCodes", [])),
        }
        for item in recommendations
    ]
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("추천 결과가 없습니다.")

    with st.expander("커리큘럼 요청·응답 JSON", expanded=False):
        st.caption("요청")
        st.json(result["request"])
        st.caption("응답")
        st.json(body)


def _render_qa_flow(
    result: dict[str, Any],
    feature_labels: dict[str, str],
) -> None:
    request = result["request"]
    response = result["response"]
    profiles = request.get("featureProfiles", [])
    recent_trainings = request.get("recentTrainings", [])
    recommendations = response.get("recommendations", [])
    if not profiles or not recommendations:
        return

    weakest = max(
        profiles,
        key=lambda profile: (
            profile.get("weaknessScore", 0),
            -profile.get("accuracyRate", 0),
        ),
    )
    weakest_code = weakest.get("featureCode", "-")
    weakest_label = feature_labels.get(weakest_code, weakest_code)
    targeted = [
        item
        for item in recommendations
        if weakest_code in item.get("targetFeatureCodes", [])
    ]
    latest_training = min(
        recent_trainings,
        key=lambda training: training.get("daysAgo", 10**9),
        default=None,
    )

    with st.container(border=True):
        st.subheader("QA 확인 흐름")
        st.caption(
            "실제 제품에서는 Backend가 훈련 결과를 저장하고 아이 프로필을 갱신합니다. "
            "이 화면은 갱신된 프로필 스냅샷이 다음 훈련 추천에 반영되는지를 확인합니다."
        )
        training_step, profile_step, recommendation_step = st.columns(
            3,
            border=True,
            vertical_alignment="top",
        )
        with training_step:
            st.markdown("**1. 훈련 결과 저장**")
            if latest_training:
                st.metric(
                    "최근 훈련 정확도",
                    f"{latest_training.get('accuracy', 0) * 100:.0f}%",
                )
                st.caption(
                    f"훈련 ID {latest_training.get('trainingTemplateId', '-')} · "
                    f"{latest_training.get('daysAgo', '-')}일 전"
                )
            else:
                st.info("최근 훈련 이력이 없습니다.")
        with profile_step:
            st.markdown("**2. 아이 프로필 갱신**")
            st.metric("가장 큰 약점", weakest_label)
            st.caption(
                f"정확도 {weakest.get('accuracyRate', 0) * 100:.0f}% · "
                f"약점 점수 {weakest.get('weaknessScore', 0) * 100:.0f}% · "
                f"근거 {weakest.get('evidenceCount', 0)}건"
            )
        with recommendation_step:
            st.markdown("**3. 다음 훈련에 반영**")
            st.metric(
                "주요 약점 대상 훈련",
                f"{len(targeted)}/{len(recommendations)}개",
            )
            for item in recommendations:
                marker = "✓" if item in targeted else "·"
                st.caption(f"{marker} {item.get('trainingName', '-')}")

        if targeted:
            st.success(
                f"갱신된 주요 약점 ‘{weakest_label}’이 다음 훈련 "
                f"{len(targeted)}개에 반영되었습니다."
            )
        else:
            st.warning(
                f"갱신된 주요 약점 ‘{weakest_label}’을 직접 대상으로 한 다음 훈련이 "
                "없습니다. 추천 규칙을 확인하세요."
            )


_initialize_state()

st.title("백엔드 프로필 AI 연동 검토")
st.caption(
    "Backend StudentFeatureProfileView 형식의 집계 프로필 하나로 교수자 분석과 "
    "다음 학습일 커리큘럼 추천을 각각 검증합니다. 원시 음성·시선 좌표·학생 식별자는 "
    "사용하지 않습니다."
)

with st.sidebar:
    st.header("API 설정")
    teacher_endpoint = st.text_input(
        "교수자 분석 API",
        value=os.getenv(
            "IREAD_TEACHER_REPORT_ENDPOINT",
            "http://127.0.0.1:8081/api/v1/reports/analyze",
        ),
        key="backend_teacher_endpoint",
    )
    curriculum_endpoint = st.text_input(
        "커리큘럼 추천 API",
        value=os.getenv(
            "IREAD_CURRICULUM_ENDPOINT",
            "http://127.0.0.1:8081/api/v1/curricula/recommend",
        ),
        key="backend_curriculum_endpoint",
    )
    api_key = st.text_input(
        "X-API-Key",
        value=_default_api_key(),
        type="password",
        key="backend_profile_api_key",
    )
    use_llm = st.toggle(
        "커리큘럼 LLM 재정렬",
        value=True,
        key="backend_profile_use_llm",
    )
    st.button(
        "익명 샘플 복원",
        icon=":material/refresh:",
        on_click=_reset_inputs,
        key="backend_profile_reset",
    )
    st.caption("API 키는 요청 헤더에만 사용하며 결과 화면에 표시하지 않습니다.")

with st.form("backend_profile_review_form", border=False):
    profile_column, context_column = st.columns(2)
    with profile_column:
        with st.container(border=True):
            st.subheader("읽기 프로필")
            profile_text = st.text_area(
                "StudentFeatureProfileView 배열",
                height=430,
                key="backend_profile_editor",
                help="status, analysisVersion, analyzedAt을 포함한 Backend 조회 형식입니다.",
            )
        with st.container(border=True):
            st.subheader("특징 이름")
            feature_labels_text = st.text_area(
                "featureCode와 featureName 매핑",
                height=180,
                key="backend_feature_labels_editor",
                help="교수자 화면에 특징 코드를 그대로 노출하지 않기 위해 사용합니다.",
            )
    with context_column:
        with st.container(border=True):
            st.subheader("시선 추세")
            gaze_trend_text = st.text_area(
                "훈련·검사 시선 추세",
                height=300,
                key="backend_gaze_trend_editor",
            )
        with st.container(border=True):
            st.subheader("최근 훈련")
            recent_trainings_text = st.text_area(
                "최근 훈련 이력 배열",
                height=300,
                key="backend_recent_trainings_editor",
            )

    with st.container(horizontal=True):
        validate_only = st.form_submit_button(
            "입력 검증",
            icon=":material/check_circle:",
        )
        run_teacher = st.form_submit_button(
            "교수자 분석",
            icon=":material/summarize:",
        )
        run_curriculum = st.form_submit_button(
            "커리큘럼 추천",
            icon=":material/route:",
        )
        run_both = st.form_submit_button(
            "둘 다 실행",
            type="primary",
            icon=":material/play_arrow:",
        )

action_requested = validate_only or run_teacher or run_curriculum or run_both
if action_requested:
    try:
        snapshot, feature_labels, gaze_trend, recent_trainings = _validated_inputs(
            profile_text,
            feature_labels_text,
            gaze_trend_text,
            recent_trainings_text,
        )
        teacher_request = build_teacher_report_request(
            request_id=f"profile-review-teacher-{uuid.uuid4().hex[:12]}",
            snapshot=snapshot,
            feature_labels=feature_labels,
            gaze_trend=gaze_trend,
        )
        curriculum_request = build_curriculum_recommend_request(
            request_id=f"profile-review-curriculum-{uuid.uuid4().hex[:12]}",
            snapshot=snapshot,
            recent_trainings=recent_trainings,
            use_llm=use_llm,
        )
    except (ValueError, ValidationError) as exception:
        st.session_state.backend_profile_validation = str(exception)
    else:
        st.session_state.backend_profile_validation = "정상"
        if run_teacher or run_both:
            try:
                st.session_state.backend_teacher_result = _post_json(
                    teacher_endpoint,
                    api_key,
                    teacher_request.model_dump(mode="json", by_alias=True),
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exception:
                st.session_state.backend_teacher_result = {"error": str(exception)}
        if run_curriculum or run_both:
            try:
                st.session_state.backend_curriculum_result = _post_json(
                    curriculum_endpoint,
                    api_key,
                    curriculum_request.model_dump(mode="json", by_alias=True),
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exception:
                st.session_state.backend_curriculum_result = {"error": str(exception)}

validation = st.session_state.backend_profile_validation
if validation == "정상":
    st.success("백엔드 프로필과 두 AI 요청 계약이 모두 유효합니다.")
elif validation:
    st.error(f"입력 검증에 실패했습니다: {validation}")

teacher_result = st.session_state.backend_teacher_result
curriculum_result = st.session_state.backend_curriculum_result
if curriculum_result and "error" not in curriculum_result:
    try:
        qa_feature_labels = _parse_json(
            st.session_state.backend_feature_labels_editor,
            label="특징 이름",
            expected_type=dict,
        )
    except ValueError:
        qa_feature_labels = {}
    _render_qa_flow(
        curriculum_result,
        {str(key): str(value) for key, value in qa_feature_labels.items()},
    )

teacher_tab, curriculum_tab = st.tabs(["교수자 분석 결과", "커리큘럼 추천 결과"])
with teacher_tab:
    if teacher_result and "error" not in teacher_result:
        _render_teacher_result(teacher_result)
    elif teacher_result:
        st.error(f"교수자 분석 요청에 실패했습니다: {teacher_result['error']}")
    else:
        st.info("입력을 검증한 뒤 교수자 분석을 실행하세요.")
with curriculum_tab:
    if curriculum_result and "error" not in curriculum_result:
        _render_curriculum_result(curriculum_result)
    elif curriculum_result:
        st.error(f"커리큘럼 추천 요청에 실패했습니다: {curriculum_result['error']}")
    else:
        st.info("입력을 검증한 뒤 커리큘럼 추천을 실행하세요.")
