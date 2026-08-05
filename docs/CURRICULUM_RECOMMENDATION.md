# 맞춤 커리큘럼 추천

`POST /api/v1/curricula/recommend`는 학생의 읽기 특징 프로필을 바탕으로 다음 학습일에
수행할 훈련 템플릿 5개와 순서를 추천한다. 문항이나 교안 내용은 생성하지 않는다.

입력 프로필은 교수자 분석 API와 같은 공통 계약을 사용한다. 정확도·취약도·신뢰도와
비율 지표는 `0.0~1.0`, 평균 발음 점수는 `0.0~100.0`이며, `featureCode`는 Backend
`reading_features.feature_code` 네임스페이스를 사용한다.

Backend `StudentFeatureProfileView`는
`reading_profile_request_adapter.build_curriculum_recommend_request`를 통해 그대로
정규화할 수 있다. 추천에 필요하지 않은 `status`, `analysisVersion`, `analyzedAt`은
제거하고, 최근 훈련 이력은 `recentTrainings`로 별도 결합한다.

## 추천 파이프라인

1. 프로필의 신뢰도와 근거 수를 확인한다.
2. 충분한 근거가 있는 특징 중 정확도 0.80 미만 또는 취약도 0.35 이상인 가장 기초
   특징을 현재 단계로 판정한다.
3. 현재 단계보다 두 단계 이상 높은 템플릿과 폐기 템플릿 `6`, `14`, `24`를 차단한다.
4. 전체 프로필은 보존하되, 근거가 충분하고 최대 허용 단계 안에 있는 특징만 현재
   학습일의 실행 가능한 약점으로 사용한다. 더 높은 단계의 큰 약점은 이후 단계 관찰
   대상으로 보류한다.
5. 템플릿의 대표 목표 특징과 정확히 일치하거나 같은 세부 특징군에 있는 프로필을 먼저
   연결한다. 직접 연결되는 특징이 없을 때만 같은 읽기 영역의 프로필을 보조 근거로 쓴다.
6. 남은 후보를 취약도, 정확도, 신뢰도, 발음, 시선 부담, 최근 반복 이력으로 점수화한다.
7. 최근 3일 안에 수행한 템플릿은 대체 가능한 후보가 충분하면 이번 추천 후보에서 제외한다.
8. 규칙 기반으로 `CORE 3 + REINFORCEMENT 1 + STRETCH 1` 기준안을 만든다.
   CORE는 현재 단계 또는 바로 이전 단계에서만 고른다.
9. GMS가 설정되어 있으면 실행 가능한 프로필 중 현재 단계와 가까운 근거 최대 12개와
   허용 후보만 전달해 5개를 재정렬한다.
10. LLM 결과의 ID, 중복, 역할 수, 최대 단계, 진단 표현을 검증한다. 또한 각 역할에서
   규칙 기준안의 최저 점수보다 낮은 후보로 교체하지 못하게 한다.
11. LLM이 없거나 검증에 실패하면 규칙 기반 기준안을 반환한다.

## 단계

| 단계 | 훈련 영역 | 대표 템플릿 |
| --- | --- | --- |
| 1 | 글자 따라 보기 | 모음·자음·음절 따라 보기 |
| 2 | 소리 듣고 고르기 | 자모·음절·받침 소리 고르기 |
| 3 | 글자 만들기 | 음절·낱말 조합 |
| 4 | 글자 자르기 | 받침·음절 빼기 |
| 5 | 글자 대치 | 음절 바꾸기 |
| 6 | 글 해독 | 낱말·문장·짧은 글 읽기 |
| 7 | 문장 완성 및 이해 | 문장 조립·빈칸·그림 연결 |
| 8 | 유창하게 읽기 | 따라 읽기·끊어 읽기·짧은 이야기 |

현재 단계가 1이면 최대 허용 단계는 2다. 따라서 `짧은 글 읽기` 26번과 8단계 유창성
훈련은 취약 특징이 일치하더라도 LLM 후보에 포함되지 않는다.

## 요청 예시

```json
{
  "requestId": "curriculum-example-1",
  "schemaVersion": 1,
  "featureProfiles": [
    {
      "featureCode": "GRAPHEME.VOWEL.BASIC.ㅏ",
      "category": "GRAPHEME",
      "accuracyRate": 0.42,
      "weaknessScore": 0.82,
      "confidence": 0.9,
      "evidenceCount": 12,
      "pronunciationErrorRate": 0.36,
      "avgFixationDurationMs": 820,
      "avgRegressionCount": 1.4,
      "skipRate": 0.08
    }
  ],
  "recentTrainings": [],
  "useLlm": true
}
```

헤더는 다음 두 개가 필요하다.

```text
X-API-Key: <AI_INTERNAL_API_KEY>
Idempotency-Key: curriculum-example-1
```

`Idempotency-Key`는 `requestId`와 같아야 한다.

## 주요 응답 필드

- `recommendationProvider`: `gms`, `deterministic`, `deterministic-fallback`
- `dataSufficiency`: `SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`
- `currentStage`: 판정된 현재 단계
- `maximumAllowedStage`: 추천할 수 있는 최대 단계
- `recommendations`: 순서가 있는 정확히 5개 훈련
- `candidateAudit`: 34개 전체 후보의 허용·차단 결과와 사유
- `warnings`: LLM fallback 또는 데이터 부족 안내

각 추천은 `trainingTemplateId`, `role`, `recommendedDifficulty`, `score`,
`targetFeatureCodes`, `reasonCodes`, `rationale`을 포함한다. 이 중
`recommendedDifficulty`는 선택된 훈련 안에서 문항을 만들 때 사용하는 1~5 난이도다.

## 로컬 실행

AI 서버를 먼저 실행한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"
.\.venv\Scripts\uvicorn.exe iread_ai.app:app --app-dir src --host 127.0.0.1 --port 8081
```

다른 PowerShell에서 검토 화면을 실행한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"
.\.venv\Scripts\streamlit.exe run curriculum_review_app.py --server.port 8509
```

브라우저에서 `http://127.0.0.1:8509`를 연다.

Docker로 검토 화면만 실행할 수도 있다.

```powershell
docker compose -f compose.curriculum-ui.yaml up --build
```

이 경우 AI 서버는 호스트의 `8081` 포트에서 실행 중이어야 한다.

## GMS와 fallback 확인

`.env`에 기존 GMS 설정을 사용한다.

```dotenv
AI_GENERATION_PROVIDER=gms
GMS_KEY=...
OPENAI_MODEL=gpt-5.4-mini
```

GMS 대신 OpenAI API를 직접 사용할 때는 `AI_GENERATION_PROVIDER=openai`와
`OPENAI_API_KEY=...`를 설정한다. 응답의 `recommendationProvider`는 실제 호출
경로에 따라 `openai` 또는 `gms`로 기록된다.

- 화면의 `GMS LLM 재정렬 사용`을 켜고 `recommendationProvider=gms`이면 LLM 결과다.
- GMS 설정이 없거나 응답 검증이 실패하면 `deterministic-fallback`이 표시된다.
- 토글을 끄면 의도적으로 `deterministic` 추천만 실행한다.

## 테스트

```powershell
.\.venv\Scripts\pytest.exe tests\unit\test_curriculum_recommender.py `
  tests\test_curriculum_api.py `
  tests\contract\test_curriculum_openapi_contract.py `
  tests\ui\test_curriculum_review_app.py
```

주요 회귀 조건은 다음과 같다.

- 자모 단계 학생에게 3단계 이상 훈련이 추천되지 않는다.
- 신규 학생은 1단계에서 시작한다.
- 추천은 항상 서로 다른 템플릿 5개다.
- 역할은 항상 `3 + 1 + 1`이다.
- 폐기 템플릿은 추천되지 않는다.
- 문장 단계 학생만 8단계 유창성 훈련을 받을 수 있다.
- LLM이 단계 제한을 넘으면 규칙 기반 결과로 fallback한다.
- 최근 3일 내 훈련은 대체 후보가 충분하면 제외하고, 후보가 부족할 때만 재사용한다.
- 고단계 약점 점수가 더 높아도 현재 최대 허용 단계의 추천 목표에는 섞이지 않는다.
- 긴 프로필을 LLM에 전달할 때 현재 단계와 가까운 근거가 최대 12개로 제한된다.
- 자음·모음·받침 훈련의 `targetFeatureCodes`가 훈련 내용과 무관한 음절 약점으로
  일괄 표시되지 않는다.
- LLM이 이미 안정적인 특징의 낮은 점수 후보를 CORE로 교체하면 규칙 기준안으로
  fallback한다.

`backend_profile_review_app.py`에서는 `긴 프로필` 버튼으로 21개 특징이 들어 있는 QA
샘플을 불러올 수 있다. 이 샘플은 문장·단어·음운 약점이 현재 음절 약점보다 크게
설정되어 있으므로, 단순 약점 순위가 아니라 선수 단계 게이트가 작동하는지 확인하는 데
사용한다.

`.env`가 실제 GMS 모드인 상태에서 전체 테스트를 실행할 때는 테스트 프로세스에만 Mock을
덮어쓴다. `.env` 파일 자체는 변경하지 않는다.

```powershell
$env:AI_GENERATION_PROVIDER='mock'
.\.venv\Scripts\pytest.exe
```
