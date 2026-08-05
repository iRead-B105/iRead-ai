# 훈련 수행 평가

## 평가 대상

`POST /api/v1/trainings/evaluate`는 생성된 문항의 품질이 아니라 학생이 수행한 훈련
결과를 평가한다. 문항 생성 품질은 `/api/v1/trainings/candidates`와
`/api/v1/training-sets/generate`의 생성·검증 단계에서 별도로 확인한다.

최종 점수는 LLM이 결정하지 않는다.

- 선택형·조립형: Backend가 기록한 `totalScore` 또는 정오답을 사용한다.
- 따라 읽기: 원본 녹음과 기준 문장을 `/api/v1/speech/pronunciation/analyze`에 전달하고,
  마지막 시도의 `pronunciationAccuracyScore`를 사용한다.
- LLM: 교수자용 설명이나 오류 경향 요약에만 사용하며 정확도를 변경하지 않는다.

평가 버전은 `IREAD_TRAINING_EVALUATION_V1`이다.

## 점수 규칙

1. `questions[].totalScore`가 있으면 Backend의 0~1000 점수를 0~100으로 환산한다.
2. `totalScore`가 없으면 `isCorrect=true` 또는 `correctionConfirmed=true`를 100점,
   `isCorrect=false`를 0점으로 처리한다.
3. `pronunciationAnalyses`는 `questionNo`별로 `attemptNo`가 가장 큰 마지막 완료 시도만
   사용한다. `questionCompleted=false`인 미완료 입력은 점수에서 제외한다.
4. 선택형 결과와 발음 분석이 같은 `questionNo`를 가지면 발음 정확도를 우선한다.
5. 문항·발음 분석이 전혀 없을 때만 최종 `wordAttempts[].totalScore`를 보조 근거로
   사용한다. 같은 `questionNo`·`targetIndex`·`tokenIndex`의 여러 기록은 마지막 확정
   시도만 사용한다.
6. 최종 정확도는 중복을 제거한 문항 점수의 산술 평균이다.
7. 채점 가능한 근거가 없으면 기존 임시 구현처럼 100점이 아니라 0점을 반환한다.
8. 점수가 허용 범위를 벗어나거나 배열·객체 구조가 잘못되면 `422`를 반환한다.

## 따라 읽기 흐름

```text
브라우저 16 kHz WAV 녹음
  → Backend 또는 검토 UI
  → AI /api/v1/speech/pronunciation/analyze
  → Azure Speech ko-KR scripted 발음평가
  → 정확도·완성도·유창성·단어별 오류와 음성 구간
  → AI /api/v1/trainings/evaluate
  → 최종 accuracy
```

운영 기본 공급자는 `AI_PRONUNCIATION_PROVIDER=azure`다. Azure scripted pronunciation
assessment가 정확도·유창성·완성도와 단어별 오류를 직접 반환한다. GMS Whisper 경로는
기존 개발 환경 호환용으로만 유지하며 운영 점수를 대신하는 fallback으로 사용하지 않는다.
Azure 장애 시 문자열 일치도로 점수를 대체하지 않고 재시도 가능한 외부 공급자 오류를
반환한다.

발음 분석 응답의 `recognizedText`에는 공급자가 인식한 문장이 들어가며 검토 화면에서는
`음성 인식 문장`으로 표시한다. 원본 녹음은 분석 요청 동안만 임시로 사용하고 저장하지
않는다.

## 서비스 적용 피드백

발음 점수는 구체적인 자음·모음 대치를 추정하는 진단값이 아니라 단어별 읽기 수행
근거로 사용한다. `training_feedback.build_pronunciation_feedback`은 발음 응답을 다음과
같이 안전하게 해석한다.

- 아이에게는 가장 낮은 단어를 최대 2개까지만 골라 다시 읽을 행동을 안내한다.
- 교수자에게는 전체 정확도·유창성·완성도와 우선 확인 단어를 수치와 함께 제공한다.
- `Omission`은 읽지 않은 단어, `Insertion`은 추가로 읽은 단어로 구분한다.
- Azure가 제공하지 않은 한국어 음소 대치나 학습장애 진단을 생성하지 않는다.
- 인식 문장은 기준 문장과 같지만 완성도가 0점인 경우 원시 결과 재확인 경고를 표시한다.
- 누적 단어 점수는 Backend가 문항의 `targetFeatureCodes`와 결합해 학생 특징 프로필,
  다음 5개 훈련 추천과 교수자 보고서 근거로 사용한다.

훈련 유형별 해석은 다음과 같다.

| 훈련 | 평가 초점 |
| --- | --- |
| 자모·음절 따라 보기 | 음성 입력 확인. 실제 아동 음성 보정 전까지 점수는 검증 자료로만 사용 |
| 글자 만들기·빼기·바꾸기 | 조작 결과를 소리 내어 읽은 정확도 |
| 낱말·문장·짧은 글 읽기 | 단어별 정확도와 누락 여부 |
| 문장 따라 읽기·끊어 읽기·반복 읽기 | 단어 정확도와 전체 유창성 |
| 짧은 이야기 읽기 | 정확도와 유창성. 한국어에서 지원하지 않는 감정·억양 점수는 생성하지 않음 |

## 평가 API 예시

```http
POST /api/v1/trainings/evaluate
Content-Type: application/json
X-API-Key: <AI_INTERNAL_API_KEY>
Idempotency-Key: training-evaluation-001
```

```json
{
  "requestId": "training-evaluation-001",
  "trainingId": 10,
  "studentId": 3,
  "trainingTemplateId": 30,
  "schemaVersion": 1,
  "result": {
    "pronunciationAnalyses": [
      {
        "questionNo": 1,
        "referenceText": "토끼가 숲길을 천천히 걸어요.",
        "pronunciationAccuracyScore": 82.5,
        "fluencyScore": 76,
        "completenessScore": 95,
        "attemptNo": 1,
        "questionCompleted": true
      }
    ]
  }
}
```

```json
{
  "requestId": "training-evaluation-001",
  "schemaVersion": 1,
  "accuracy": 82.5
}
```

응답 헤더 `X-AI-Provider`는 `hybrid-evaluator`이다. 같은 멱등키와 같은 요청을 다시
보내면 `Idempotent-Replayed: true`가 추가된다.

## 직접 녹음 검토 화면

AI API가 8081 포트에서 실행 중일 때 다음 명령으로 화면을 실행한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"
& ".\.venv\Scripts\streamlit.exe" run .\training_evaluation_review_app.py `
  --server.address 127.0.0.1 --server.port 8511
```

브라우저에서 `http://127.0.0.1:8511`을 연다.

- `샘플 결과`: 녹음 없이 객관식·부분 점수·발음 재시도·혼합 훈련을 검증한다.
- `직접 녹음`: Chrome에서는 마이크로 바로 녹음할 수 있고, Codex 내장 브라우저에서는
  WAV·MP3·M4A·MP4·WebM·OGG 녹음 파일을 업로드해 같은 평가 흐름을 실행할 수 있다.

실제 훈련 평가는 Azure 설정을 사용한다.

```dotenv
AI_PRONUNCIATION_PROVIDER=azure
AZURE_SPEECH_KEY=발급받은_키
AZURE_SPEECH_REGION=koreacentral
AZURE_SPEECH_LANGUAGE=ko-KR
```

Docker UI는 다음 명령으로 실행할 수 있다.

```powershell
docker compose -f compose.training-evaluation-ui.yaml up -d --build
```

## 검증

```powershell
& ".\.venv\Scripts\pytest.exe" `
  tests\unit\test_training_feedback.py `
  tests\unit\test_training_evaluation.py `
  tests\test_training_evaluation_api.py `
  tests\ui\test_training_evaluation_review_app.py
```
