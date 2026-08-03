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
3. `pronunciationAnalyses`는 `questionNo`별로 `attemptNo`가 가장 큰 마지막 시도만
   사용한다.
4. 선택형 결과와 발음 분석이 같은 `questionNo`를 가지면 발음 정확도를 우선한다.
5. 문항·발음 분석이 전혀 없을 때만 최종 `wordAttempts[].totalScore`를 보조 근거로
   사용한다.
6. 최종 정확도는 중복을 제거한 문항 점수의 산술 평균이다.
7. 채점 가능한 근거가 없으면 기존 임시 구현처럼 100점이 아니라 0점을 반환한다.
8. 점수가 허용 범위를 벗어나거나 배열·객체 구조가 잘못되면 `422`를 반환한다.

## 따라 읽기 흐름

```text
브라우저 16 kHz WAV 녹음
  → Backend 또는 검토 UI
  → AI /api/v1/speech/pronunciation/analyze
  → Azure 발음평가 또는 GMS Whisper 전사 기반 읽기 일치도
  → 정확도·완성도·단어별 오류(Azure는 유창성 포함)
  → AI /api/v1/trainings/evaluate
  → 최종 accuracy
```

`AI_PRONUNCIATION_PROVIDER=gms`는 녹음을 GMS `whisper-1`로 실제 전사한 뒤 기준 문장과
한글 자모 단위 편집 거리를 계산한다. 따라서 녹음 내용에 따라 점수가 달라지지만,
음소·강세·운율을 직접 평가하는 임상적 발음 점수는 아니다. Azure를 사용할 수 있을 때는
scripted pronunciation assessment가 정확도·유창성·완성도를 직접 반환한다.
발음 분석 응답의 `recognizedText`에는 GMS Whisper 또는 Azure가 실제로 인식한 문장이
들어가며, 검토 화면에서는 기준 문장과 함께 표시한다.

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

GMS 키가 이미 설정된 개발 환경에서는 다음 설정으로 실제 녹음 기반 읽기 일치도를 쓸 수
있다.

```dotenv
AI_PRONUNCIATION_PROVIDER=gms
AI_GMS_SPEECH_MODEL=whisper-1
AI_GMS_SPEECH_TIMEOUT_SECONDS=45
```

음소·유창성까지 평가하려면 Azure 설정을 사용한다.

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
  tests\unit\test_training_evaluation.py `
  tests\test_training_evaluation_api.py `
  tests\ui\test_training_evaluation_review_app.py
```
