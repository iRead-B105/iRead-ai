from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from iread_ai.devtools.training_evaluation_samples import (
    READ_ALOUD_SENTENCES,
    TRAINING_EVALUATION_SAMPLES,
    training_evaluation_sample,
)
from iread_ai.training_feedback import (
    TrainingFeedbackError,
    build_pronunciation_feedback,
)

load_dotenv()

st.set_page_config(
    page_title="iRead 훈련 평가 검토",
    page_icon=":material/record_voice_over:",
    layout="wide",
)

st.session_state.setdefault("training_evaluation_response", None)
st.session_state.setdefault("pronunciation_response", None)
st.session_state.setdefault("training_evaluation_request", None)
st.session_state.setdefault("training_evaluation_elapsed_ms", None)


def _api_key() -> str:
    return (
        os.getenv("AI_INTERNAL_API_KEY", "").strip()
        or os.getenv("AI_API_KEY", "").strip()
        or "local-development-key"
    )


def _post_json(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float, str]:
    started = time.perf_counter()
    with httpx.Client(timeout=40) as client:
        response = client.post(
            endpoint,
            headers={
                "X-API-Key": api_key,
                "Idempotency-Key": payload["requestId"],
                "Content-Type": "application/json",
            },
            json=payload,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    return response.json(), elapsed_ms, response.headers.get("X-AI-Provider", "")


def _post_pronunciation(
    endpoint: str,
    api_key: str,
    request_id: str,
    expected_text: str,
    audio_name: str,
    audio_bytes: bytes,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with httpx.Client(timeout=45) as client:
        response = client.post(
            endpoint,
            headers={
                "X-API-Key": api_key,
                "Idempotency-Key": request_id,
            },
            data={"requestId": request_id, "expectedText": expected_text},
            files={"audioFile": (audio_name, audio_bytes, "audio/wav")},
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    return response.json(), elapsed_ms


TRAINING_TYPES = {
    "낱말 읽기": 22,
    "문장 읽기": 25,
    "짧은 글 읽기": 26,
    "문장 따라 읽기": 30,
    "단어 이어 읽기": 31,
    "끊어 읽기": 32,
    "같은 문장 다시 읽기": 33,
    "짧은 이야기 읽기": 34,
}


def _evaluation_payload(
    result: dict[str, Any],
    *,
    training_template_id: int = 30,
) -> dict[str, Any]:
    request_id = f"training-evaluation-review-{uuid.uuid4().hex[:12]}"
    return {
        "requestId": request_id,
        "trainingId": 1,
        "studentId": 1,
        "trainingTemplateId": training_template_id,
        "schemaVersion": 1,
        "result": result,
    }


def _pronunciation_result(analysis: dict[str, Any], expected_text: str) -> dict[str, Any]:
    return {
        "pronunciationAnalyses": [
            {
                "questionNo": 1,
                "referenceText": expected_text,
                "pronunciationAccuracyScore": analysis["pronunciationAccuracyScore"],
                "fluencyScore": analysis.get("fluencyScore"),
                "completenessScore": analysis.get("completenessScore"),
                "pronScore": analysis.get("pronScore"),
                "confidence": analysis.get("confidence"),
                "analysisVersion": analysis.get("analysisVersion"),
                "attemptNo": 1,
                "passed": analysis["pronunciationAccuracyScore"] >= 70,
                "questionCompleted": True,
            }
        ]
    }


def _clear_evaluation_result() -> None:
    st.session_state.training_evaluation_response = None
    st.session_state.pronunciation_response = None
    st.session_state.training_evaluation_request = None
    st.session_state.training_evaluation_elapsed_ms = None


st.title("훈련 평가 검토")
st.caption(
    "선택형 문항은 결정적 규칙으로 채점하고, 따라 읽기는 원본 녹음과 기준 문장을 "
    "발음 평가 모델에 전달한 뒤 구조화된 정확도를 최종 훈련 점수에 반영합니다."
)

with st.sidebar:
    st.header("API 설정")
    evaluation_endpoint = st.text_input(
        "훈련 평가 API",
        value=os.getenv(
            "IREAD_TRAINING_EVALUATION_ENDPOINT",
            "http://127.0.0.1:8081/api/v1/trainings/evaluate",
        ),
        key="training_evaluation_endpoint",
    )
    pronunciation_endpoint = st.text_input(
        "발음 분석 API",
        value=os.getenv(
            "IREAD_PRONUNCIATION_ENDPOINT",
            "http://127.0.0.1:8081/api/v1/speech/pronunciation/analyze",
        ),
        key="training_pronunciation_endpoint",
    )
    api_key = st.text_input(
        "X-API-Key",
        value=_api_key(),
        type="password",
        key="training_evaluation_api_key",
    )
    st.caption("API 키와 녹음 파일은 결과 화면에 표시하지 않습니다.")

mode = st.segmented_control(
    "검토 방식",
    ["샘플 결과", "직접 녹음"],
    default="샘플 결과",
    key="training_evaluation_mode",
    on_change=_clear_evaluation_result,
)

if mode == "샘플 결과":
    sample_name = st.selectbox(
        "샘플",
        list(TRAINING_EVALUATION_SAMPLES),
        key="training_evaluation_sample",
    )
    sample = training_evaluation_sample(sample_name)
    st.info(sample["description"])
    with st.expander("평가 입력 JSON", expanded=False):
        st.json(sample["result"])

    if st.button(
        "샘플 평가 실행",
        type="primary",
        icon=":material/analytics:",
        key="evaluate_sample",
    ):
        payload = _evaluation_payload(sample["result"])
        try:
            result, elapsed_ms, provider = _post_json(
                evaluation_endpoint,
                api_key,
                payload,
            )
        except (httpx.HTTPError, ValueError, KeyError) as exception:
            st.error(f"평가 요청에 실패했습니다: {exception}")
        else:
            st.session_state.training_evaluation_response = result
            st.session_state.pronunciation_response = None
            st.session_state.training_evaluation_request = payload
            st.session_state.training_evaluation_elapsed_ms = elapsed_ms
            st.session_state.training_evaluation_provider = provider
else:
    provider_mode = os.getenv("AI_PRONUNCIATION_PROVIDER", "deterministic").strip().lower()
    pronunciation_ready = provider_mode in {"azure", "gms"}
    if provider_mode == "gms":
        st.info(
            "현재 GMS Whisper가 녹음을 실제로 전사하고, 기준 문장과 한글 자모 단위로 "
            "비교해 읽기 일치도를 계산합니다. 유창성·운율 점수는 제공하지 않습니다."
        )
    elif provider_mode == "azure":
        st.info("현재 Azure Speech가 정확도·유창성·완성도와 단어별 발음 오류를 평가합니다.")
    else:
        st.error(
            "현재 deterministic 모드는 실제 음성을 분석하지 않으므로 녹음 평가를 실행할 수 "
            "없습니다. AI_PRONUNCIATION_PROVIDER를 gms 또는 azure로 설정하세요."
        )

    selected_training_type = st.selectbox(
        "훈련 유형",
        list(TRAINING_TYPES),
        index=3,
        key="pronunciation_training_type",
        help="훈련 유형에 따라 단어 정확도와 유창성의 해석 기준을 구분합니다.",
    )
    selected_training_template_id = TRAINING_TYPES[selected_training_type]
    selected_sentence = st.selectbox(
        "따라 읽을 문장",
        READ_ALOUD_SENTENCES,
        key="read_aloud_sentence",
    )
    expected_text = st.text_input(
        "기준 문장",
        value=selected_sentence,
        key="read_aloud_expected_text",
        help="녹음에서 읽을 문장과 정확히 같아야 합니다.",
    )
    audio_input_method = st.segmented_control(
        "음성 입력 방식",
        ["파일 업로드", "마이크 녹음"],
        default="파일 업로드",
        key="training_audio_input_method",
        on_change=_clear_evaluation_result,
    )
    with st.container(border=True):
        st.markdown(f"**읽어 주세요:** {expected_text}")
        if audio_input_method == "마이크 녹음":
            audio_file = st.audio_input(
                "목소리 녹음",
                sample_rate=16000,
                key="training_voice_recording",
                help="조용한 곳에서 문장을 한 번 자연스럽게 읽어 주세요.",
                disabled=not pronunciation_ready,
            )
        else:
            st.caption(
                "Codex 내장 브라우저에서는 외부에서 녹음한 파일을 선택해 주세요. "
                "최대 20MB까지 업로드할 수 있습니다."
            )
            audio_file = st.file_uploader(
                "녹음 파일",
                type=["wav", "mp3", "m4a", "mp4", "webm", "ogg"],
                key="training_voice_file",
                help="WAV, MP3, M4A, MP4, WebM, OGG 파일을 지원합니다.",
                max_upload_size=20,
                disabled=not pronunciation_ready,
            )
        if audio_file is not None:
            st.audio(audio_file)

    if st.button(
        "음성 분석 및 훈련 평가",
        type="primary",
        icon=":material/mic:",
        key="evaluate_recording",
        disabled=not pronunciation_ready,
    ):
        if audio_file is None:
            st.warning("먼저 목소리를 녹음하거나 녹음 파일을 업로드해 주세요.")
        elif not expected_text.strip():
            st.warning("기준 문장을 입력해 주세요.")
        else:
            pronunciation_request_id = f"pronunciation-review-{uuid.uuid4().hex[:12]}"
            try:
                with st.spinner("녹음을 분석하고 최종 점수를 계산하는 중입니다..."):
                    pronunciation, pronunciation_ms = _post_pronunciation(
                        pronunciation_endpoint,
                        api_key,
                        pronunciation_request_id,
                        expected_text.strip(),
                        audio_file.name or "recording.wav",
                        audio_file.getvalue(),
                    )
                    payload = _evaluation_payload(
                        _pronunciation_result(pronunciation, expected_text.strip()),
                        training_template_id=selected_training_template_id,
                    )
                    result, evaluation_ms, provider = _post_json(
                        evaluation_endpoint,
                        api_key,
                        payload,
                    )
            except (httpx.HTTPError, ValueError, KeyError) as exception:
                st.error(f"녹음 평가에 실패했습니다: {exception}")
            else:
                st.session_state.training_evaluation_response = result
                st.session_state.pronunciation_response = pronunciation
                st.session_state.training_evaluation_request = payload
                st.session_state.training_evaluation_elapsed_ms = round(
                    pronunciation_ms + evaluation_ms,
                    1,
                )
                st.session_state.training_evaluation_provider = provider

evaluation = st.session_state.training_evaluation_response
pronunciation = st.session_state.pronunciation_response

if evaluation is not None:
    st.success("훈련 평가가 완료되었습니다.")
    with st.container(border=True):
        metrics = st.columns(4)
        metrics[0].metric("최종 정확도", f"{evaluation['accuracy']:.2f}점")
        metrics[1].metric(
            "발음 정확도",
            (
                f"{pronunciation['pronunciationAccuracyScore']:.2f}점"
                if pronunciation is not None
                else "해당 없음"
            ),
        )
        metrics[2].metric(
            "유창성",
            (
                f"{pronunciation.get('fluencyScore'):.2f}점"
                if pronunciation is not None and pronunciation.get("fluencyScore") is not None
                else "해당 없음"
            ),
        )
        metrics[3].metric(
            "처리 시간",
            f"{st.session_state.training_evaluation_elapsed_ms:.1f}ms",
        )
        evaluation_provider = (
            st.session_state.get("training_evaluation_provider") or "hybrid-evaluator"
        )
        st.caption(f"평가 방식: {evaluation_provider}")

    if pronunciation is not None:
        recognized_text = pronunciation.get("recognizedText")
        reference_text = st.session_state.training_evaluation_request["result"][
            "pronunciationAnalyses"
        ][0]["referenceText"]
        with st.container(border=True):
            st.markdown(f"**기준 문장**  \n{reference_text}")
            if recognized_text:
                st.markdown(f"**음성 인식 문장**  \n{recognized_text}")
            else:
                st.caption("현재 발음 분석 provider는 인식 문장을 제공하지 않습니다.")

        training_template_id = st.session_state.training_evaluation_request["trainingTemplateId"]
        try:
            feedback = build_pronunciation_feedback(
                pronunciation,
                reference_text=reference_text,
                training_template_id=training_template_id,
            )
        except TrainingFeedbackError as exception:
            st.warning(f"서비스 피드백을 구성하지 못했습니다: {exception}")
        else:
            st.subheader("서비스 적용 피드백")
            with st.container(border=True):
                st.caption(f"평가 초점: {feedback.evaluation_focus}")
                if feedback.retry_recommended:
                    st.warning(feedback.child_summary, icon=":material/replay:")
                else:
                    st.success(feedback.child_summary, icon=":material/check_circle:")
                if feedback.focus_words:
                    st.write("우선 연습 단어: " + ", ".join(feedback.focus_words))
                if feedback.strengths:
                    st.write(" ".join(feedback.strengths))
                for caution in feedback.cautions:
                    st.caption(f"확인 필요: {caution}")

            with st.container(border=True):
                st.markdown("**교수자 관찰 문장**")
                st.write(feedback.teacher_observation)
                st.caption(
                    "단일 녹음으로 자음·모음 대치나 학습장애를 진단하지 않습니다. "
                    "누적 단어 점수는 문항의 읽기 특징 코드와 결합해 학생 프로필 근거로 사용합니다."
                )

        st.subheader("단어별 발음 결과")
        if "feedback" in locals():
            rows = [
                {
                    "단어": item.word,
                    "정확도": item.score,
                    "판정": item.label,
                    "서비스 피드백": item.guidance,
                    "시작(ms)": item.offset_ms,
                    "종료(ms)": item.end_ms,
                }
                for item in feedback.words
            ]
        else:
            rows = [
                {
                    "단어": item["word"],
                    "정확도": item.get("accuracyScore"),
                    "판정": item["errorType"],
                    "서비스 피드백": "결과를 다시 확인해 주세요.",
                    "시작(ms)": item["offsetMs"],
                    "종료(ms)": item["offsetMs"] + item["durationMs"],
                }
                for item in pronunciation["words"]
            ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "정확도": st.column_config.ProgressColumn(
                    "정확도",
                    min_value=0,
                    max_value=100,
                    format="%.1f",
                )
            },
            key="pronunciation_words",
        )

    with st.expander("요청·응답 JSON 확인", expanded=False):
        st.markdown("**훈련 평가 요청**")
        st.json(st.session_state.training_evaluation_request)
        if pronunciation is not None:
            st.markdown("**발음 분석 응답**")
            st.json(pronunciation)
        st.markdown("**훈련 평가 응답**")
        st.json(evaluation)
