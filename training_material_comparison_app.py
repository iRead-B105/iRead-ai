from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iread_ai.devtools.training_comparison_data import (  # noqa: E402
    AI_BEGINNER_PROFILE,
    ORIGINAL_MOCKS,
    TRAINING_TYPES,
    beginner_targets,
)
from iread_ai.devtools.training_review_catalog import output_template  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="AI 기초형 훈련 34종 비교",
    page_icon=":material/fact_check:",
    layout="wide",
)

ERROR_CATEGORIES = (
    "목표 특성 미반영",
    "문항과 정답 불일치",
    "API 형식 오류",
    "난이도 부적절",
    "부자연스럽거나 잘못된 낱말",
    "문법 또는 문장 오류",
    "후보 중복·다양성 부족",
    "기타",
)

ANSWER_KEYS = {
    "answerIndex",
    "answerOrder",
    "result",
    "initialAnswerIndex",
    "medialAnswerIndex",
    "finalAnswerIndex",
    "deleteIndex",
    "replaceIndex",
    "completedSentence",
    "expectedText",
}

CATALOG = {
    training_type: {
        "templateId": template_id,
        "group": group,
        "name": name,
        "trainingType": training_type,
    }
    for template_id, group, name, training_type in TRAINING_TYPES
}

st.session_state.setdefault("comparison_results", {})
st.session_state.setdefault("comparison_reviews", {})
st.session_state.setdefault("comparison_errors", {})


def _api_key() -> str:
    return (
        os.getenv("AI_INTERNAL_API_KEY", "").strip()
        or os.getenv("AI_API_KEY", "").strip()
        or "local-development-key"
    )


@st.cache_data
def _backend_templates() -> dict[str, dict[str, Any]]:
    path = ROOT_DIR.parent / "backend" / "src" / "main" / "resources" / "training-templates.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["prompt"]["trainingType"]: item for item in payload.get("templates", [])}


def _request_payload(training_type: str) -> dict[str, Any]:
    request_id = f"beginner-review-{training_type.lower()}-{uuid.uuid4().hex[:10]}"
    targets = beginner_targets(training_type)
    target_names = ", ".join(target["featureCode"] for target in targets)
    profile_text = target_names or "현재 측정된 호환 목표 특성 없음"
    template = _backend_templates().get(training_type, {})
    prompt = template.get("prompt", {})
    contract_instruction = prompt.get("additionalPrompt", "")
    generated_template = prompt.get("outputTemplate", output_template(training_type))
    return {
        "requestId": request_id,
        "schemaVersion": 2,
        "trainingType": training_type,
        "count": 5,
        "difficulty": 1,
        "targetFeatures": targets,
        "excludedFeatures": [],
        "additionalPrompt": (
            f"{contract_instruction}\n\n"
            "AI 기초형 아동에게 제시할 서로 다른 문항 5개를 생성하세요. "
            f"이번 훈련의 호환 목표는 {profile_text}입니다. "
            "정답이 하나로 명확하고 초급 아동이 수행할 수 있어야 합니다."
        ),
        "outputTemplate": generated_template,
        "useLexicon": True,
    }


def _generate(training_type: str, endpoint: str, api_key: str) -> dict[str, Any]:
    payload = _request_payload(training_type)
    started = time.perf_counter()
    with httpx.Client(timeout=90) as client:
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
    body = response.json()
    if body.get("type") != training_type or len(body.get("data", [])) != 5:
        raise ValueError("AI 서버가 요청한 유형의 후보 5개를 반환하지 않았습니다.")
    return {
        "request": payload,
        "response": body,
        "provider": response.headers.get("X-AI-Provider", "unknown"),
        "fallback": response.headers.get("X-AI-Fallback"),
        "elapsedMs": elapsed_ms,
        "generatedAt": datetime.now(UTC).isoformat(),
    }


def _generate_one(training_type: str, endpoint: str, api_key: str) -> None:
    try:
        st.session_state.comparison_results[training_type] = _generate(
            training_type,
            endpoint,
            api_key,
        )
        st.session_state.comparison_errors.pop(training_type, None)
    except (httpx.HTTPError, ValueError) as exception:
        st.session_state.comparison_errors[training_type] = str(exception)
        raise


def _review_status(training_type: str) -> str:
    if training_type in st.session_state.comparison_errors:
        return "생성 실패"
    if training_type not in st.session_state.comparison_results:
        return "미생성"
    review = st.session_state.comparison_reviews.get(training_type)
    if not review:
        return "검토 대기"
    return str(review["status"])


def _split_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = {key: value for key, value in candidate.items() if key in ANSWER_KEYS}
    content = {key: value for key, value in candidate.items() if key not in ANSWER_KEYS}
    return content, answer


def _export_record() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "profile": {
            "studentId": 2115,
            "name": "AI 기초형",
            "difficulty": 1,
            "assessedFeatureCount": len(AI_BEGINNER_PROFILE),
        },
        "exportedAt": datetime.now(UTC).isoformat(),
        "items": [
            {
                **CATALOG[training_type],
                "originalMock": ORIGINAL_MOCKS[training_type],
                "generated": st.session_state.comparison_results.get(training_type),
                "review": st.session_state.comparison_reviews.get(training_type),
                "generationError": st.session_state.comparison_errors.get(training_type),
            }
            for _, _, _, training_type in TRAINING_TYPES
        ],
    }


def _save_local_report() -> Path:
    output_dir = ROOT_DIR / "local-output" / "training-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"ai-beginner-training-34-review-{stamp}.json"
    path.write_text(
        json.dumps(_export_record(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


st.title("AI 기초형 · 훈련 34종 목데이터 비교")
st.caption(
    "교수자 화면의 원래 기본 목데이터와 AI 기초형 프로필을 적용한 AI 서버 후보를 "
    "훈련별로 비교하고, 잘못된 후보를 표시하는 개발 검수 도구입니다."
)

with st.sidebar:
    st.header("생성 설정")
    endpoint = st.text_input(
        "AI 후보 생성 API",
        value=os.getenv(
            "IREAD_TRAINING_ENDPOINT",
            "http://127.0.0.1:8081/api/v1/trainings/candidates",
        ),
    )
    api_key = st.text_input("X-API-Key", value=_api_key(), type="password")
    st.text_input("프로필", value="AI 기초형 · studentId 2115", disabled=True)
    st.text_input("난이도", value="1 / 5", disabled=True)
    st.caption(
        "후보 API는 현재 AI 서버 설정에 따라 OpenAI를 호출할 수 있습니다. "
        "각 결과의 실제 provider를 화면에 표시합니다."
    )
    group_filter = st.selectbox(
        "영역 필터",
        ["전체", *dict.fromkeys(group for _, group, _, _ in TRAINING_TYPES)],
    )
    status_filter = st.selectbox(
        "검수 상태",
        ["전체", "미생성", "검토 대기", "정상", "수정 필요", "보류", "생성 실패"],
    )

status_counts = {
    status: sum(
        _review_status(training_type) == status for _, _, _, training_type in TRAINING_TYPES
    )
    for status in ("미생성", "검토 대기", "정상", "수정 필요", "보류", "생성 실패")
}

with st.container(horizontal=True):
    st.metric("전체", "34종")
    st.metric("생성 완료", f"{34 - status_counts['미생성'] - status_counts['생성 실패']}종")
    st.metric("정상", f"{status_counts['정상']}종")
    st.metric("수정 필요", f"{status_counts['수정 필요']}종")
    st.metric("보류·실패", f"{status_counts['보류'] + status_counts['생성 실패']}종")

with st.container(border=True):
    st.subheader("34종 일괄 생성")
    st.write(
        "미생성 유형만 순서대로 생성합니다. 최대 34회 API 요청이며, "
        "현재 설정에서는 각 유형이 OpenAI를 호출할 수 있습니다."
    )
    if st.button(
        "미생성 34종 생성 시작",
        type="primary",
        icon=":material/auto_awesome:",
        disabled=status_counts["미생성"] == 0,
    ):
        pending = [
            training_type
            for _, _, _, training_type in TRAINING_TYPES
            if training_type not in st.session_state.comparison_results
        ]
        progress = st.progress(0, text="생성을 준비합니다.")
        failures: list[str] = []
        for index, training_type in enumerate(pending, start=1):
            metadata = CATALOG[training_type]
            progress.progress(
                (index - 1) / len(pending),
                text=f"{metadata['templateId']}. {metadata['name']} 생성 중",
            )
            try:
                _generate_one(training_type, endpoint, api_key)
            except (httpx.HTTPError, ValueError):
                failures.append(training_type)
            progress.progress(
                index / len(pending),
                text=f"{index}/{len(pending)} 처리 완료",
            )
        if failures:
            st.warning(f"{len(failures)}개 유형이 실패했습니다: {', '.join(failures)}")
        else:
            st.success("34종 후보 생성이 끝났습니다.")
        st.rerun()

visible = [
    training_type
    for _, group, _, training_type in TRAINING_TYPES
    if (group_filter == "전체" or group == group_filter)
    and (status_filter == "전체" or _review_status(training_type) == status_filter)
]

if not visible:
    st.info("선택한 조건에 맞는 훈련이 없습니다.")
    st.stop()

selected_type = st.selectbox(
    "상세 비교할 훈련",
    visible,
    format_func=lambda training_type: (
        f"{CATALOG[training_type]['templateId']}. {CATALOG[training_type]['name']} "
        f"· {_review_status(training_type)}"
    ),
)
metadata = CATALOG[selected_type]
targets = beginner_targets(selected_type)

with st.container(border=True):
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        st.subheader(f"{metadata['templateId']}. {metadata['name']}")
        st.badge(_review_status(selected_type))
    st.caption(f"{metadata['group']} · `{selected_type}`")
    if targets:
        st.write("AI 기초형 목표: " + ", ".join(target["featureCode"] for target in targets))
    else:
        st.warning(
            "AI 기초형 프로필에 이 훈련과 호환되는 측정 특성이 없습니다. "
            "난이도 1의 일반 문항으로 생성됩니다."
        )
    if st.button(
        "이 유형 후보 5개 생성·재생성",
        icon=":material/refresh:",
        key=f"generate-{selected_type}",
    ):
        try:
            with st.spinner("AI 서버에서 후보를 생성하고 있습니다..."):
                _generate_one(selected_type, endpoint, api_key)
            st.rerun()
        except (httpx.HTTPError, ValueError) as exception:
            st.error(f"생성에 실패했습니다: {exception}")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("원래 프론트 목데이터")
    st.caption("교수자 교안 편집 화면의 기본값 1개")
    original = ORIGINAL_MOCKS[selected_type]
    with st.container(border=True):
        st.markdown("**화면에 제시되는 문항**")
        st.json(original["content"], expanded=True)
        st.markdown("**정답 데이터**")
        st.json(original["answer"], expanded=True)

with right:
    st.subheader("AI 기초형 생성 데이터")
    record = st.session_state.comparison_results.get(selected_type)
    if record:
        with st.container(horizontal=True):
            st.metric("Provider", record["provider"])
            st.metric("응답 시간", f"{record['elapsedMs'] / 1000:.2f}초")
        if record.get("fallback"):
            st.warning(f"Fallback: {record['fallback']}")
        invalid_default = st.session_state.comparison_reviews.get(selected_type, {}).get(
            "invalidCandidates",
            [],
        )
        for index, candidate in enumerate(record["response"]["data"], start=1):
            title = f"후보 {index}"
            if index in invalid_default:
                title += " · 오류 표시됨"
            with st.expander(title, expanded=index == 1):
                content, answer = _split_candidate(candidate)
                st.markdown("**문항 데이터**")
                st.json(content, expanded=True)
                if answer:
                    st.markdown("**정답·검증 데이터**")
                    st.json(answer, expanded=True)
        with st.expander("AI 서버에 보낸 요청과 원본 응답"):
            st.markdown("**요청**")
            st.json(record["request"], expanded=False)
            st.markdown("**응답**")
            st.json(record["response"], expanded=False)
    elif selected_type in st.session_state.comparison_errors:
        st.error(st.session_state.comparison_errors[selected_type])
    else:
        st.info("아직 생성 결과가 없습니다. 이 유형만 생성하거나 34종 일괄 생성을 실행하세요.")

if record:
    st.subheader("검수 결과 입력")
    previous = st.session_state.comparison_reviews.get(selected_type, {})
    with st.form(f"review-form-{selected_type}", border=True):
        status = st.segmented_control(
            "유형 전체 판정",
            ["정상", "수정 필요", "보류"],
            default=previous.get("status", "보류"),
            selection_mode="single",
        )
        invalid_candidates = st.multiselect(
            "잘못된 후보 번호",
            [1, 2, 3, 4, 5],
            default=previous.get("invalidCandidates", []),
            help="여러 개를 선택할 수 있습니다.",
        )
        categories = st.multiselect(
            "문제 유형",
            ERROR_CATEGORIES,
            default=previous.get("errorCategories", []),
        )
        note = st.text_area(
            "구체적인 문제와 수정 방향",
            value=previous.get("note", ""),
            placeholder="예: 후보 2의 answerIndex가 실제 정답 위치와 다름",
        )
        if st.form_submit_button("이 검수 결과 저장", type="primary"):
            st.session_state.comparison_reviews[selected_type] = {
                "status": status or "보류",
                "invalidCandidates": invalid_candidates,
                "errorCategories": categories,
                "note": note.strip(),
                "reviewedAt": datetime.now(UTC).isoformat(),
            }
            st.rerun()

st.divider()
st.subheader("34종 전체 현황")
rows = []
for template_id, group, name, training_type in TRAINING_TYPES:
    result = st.session_state.comparison_results.get(training_type)
    review = st.session_state.comparison_reviews.get(training_type, {})
    rows.append(
        {
            "번호": template_id,
            "영역": group,
            "훈련": name,
            "AI 방식": result["provider"] if result else "-",
            "호환 목표 수": len(beginner_targets(training_type)),
            "상태": _review_status(training_type),
            "오류 후보": ", ".join(map(str, review.get("invalidCandidates", []))) or "-",
            "메모": review.get("note", ""),
        }
    )
st.dataframe(rows, hide_index=True, width="stretch")

export_json = json.dumps(_export_record(), ensure_ascii=False, indent=2)
with st.container(horizontal=True):
    st.download_button(
        "전체 비교·검수 JSON 다운로드",
        data=export_json,
        file_name="ai-beginner-training-34-review.json",
        mime="application/json",
        icon=":material/download:",
    )
    if st.button("로컬 파일로 저장", icon=":material/save:"):
        saved_path = _save_local_report()
        st.success(f"저장했습니다: {saved_path}")
