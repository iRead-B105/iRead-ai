# iRead AI

iRead의 FastAPI 기반 AI 서비스입니다. GMS 맞춤 훈련 생성, 10일 분기 이야기와
Azure Speech 기반 발음 평가·STT·TTS를 제공합니다.

## 제공 엔드포인트

| Method | Path | 역할 |
|---|---|---|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/api/v1/trainings/candidates` | 34개 훈련 타입별 문항 후보 생성 |
| `POST` | `/api/v1/trainings/generate` | 레거시 훈련 데이터 envelope 생성 |
| `POST` | `/api/v1/trainings/evaluate` | 훈련 결과 정확도 평가 |
| `POST` | `/api/v1/story/generate` | 1일차 첫 4페이지와 첫 분기 생성 |
| `POST` | `/api/v1/story/continue` | 음성 선택을 반영해 당일 5페이지·마감 1페이지 또는 다음 날 첫 4페이지 생성 |
| `POST` | `/api/v1/images/generate` | 훈련 장면·이야기 친구 이미지 URL 생성 |
| `GET` | `/api/v1/images/mock/generated.svg` | 생성된 mock SVG 조회 |
| `POST` | `/api/v1/speech/pronunciation/analyze` | Azure 단어별 발음 평가 |
| `POST` | `/api/v1/speech/transcribe` | 이야기·훈련 음성 STT |
| `POST` | `/api/v1/speech/synthesize` | 이야기 문장 TTS |

정확한 요청·응답 모델은 실행 후 `http://localhost:8081/docs`의 OpenAPI
문서에서 확인할 수 있습니다.

## 실행

백엔드 저장소의 Docker Compose로 실행하는 방법을 권장합니다.

```powershell
# iRead-backend
docker compose up -d --build ai-mock
```

이 경우 AI 서버는 `http://localhost:8081`에서 실행됩니다. Python으로 직접
실행할 때는 8081을 사용하는 기존 `ai-mock` 컨테이너를 먼저 중지합니다.

```powershell
Copy-Item .env.example .env
uv sync --extra dev
uv run uvicorn iread_ai.app:app --host 0.0.0.0 --port 8081
uv run pytest
```

Backend의 `AI_API_KEY`와 AI 서버의 `AI_INTERNAL_API_KEY`는 같은 값을
사용합니다.

## GMS 맞춤 훈련 생성

훈련 후보는 GMS의 OpenAI 호환 Responses API와 `gpt-5.4-mini`를 사용합니다.

- 자동 테스트와 기본 로컬 실행: `AI_GENERATION_PROVIDER=mock`
- 실제 GMS 실행: `AI_GENERATION_PROVIDER=gms`와 `GMS_KEY` 설정
- 외부 응답은 JSON Schema와 훈련별 안전 규칙을 통과해야 사용
- 공급자 오류·시간 초과·검증 실패 시 안전한 결정적 후보로 대체
- 같은 멱등성 키와 본문은 저장 응답을 재생하고, 다른 본문은 `409` 반환
- Production에서는 기본 개발 키와 Mock 생성 provider 사용 금지

## 발음 평가

`POST /api/v1/speech/pronunciation/analyze`는 30초 미만 한국어 읽기 녹음과
기준 문장을 받아 Azure Speech scripted Pronunciation Assessment를 실행합니다.

- locale: `ko-KR`
- grading system: `HundredMark`
- granularity: `Word`
- miscue: 활성화
- 음성은 임시 파일로만 사용하고 요청 종료 시 삭제
- Azure 자격증명과 원본 응답은 응답·로그에 노출하지 않음

WAV는 8/16kHz, 16-bit mono PCM을 기본 입력으로 사용합니다. WebM·MP3·MP4/M4A
같은 압축 음성 처리를 위해서는 Azure Speech SDK와 같은 아키텍처의 GStreamer
런타임 및 플러그인이 필요합니다.

## 생성 mock 책임 범위

생성 mock은 동일한 입력에 동일한 훈련 문항·이야기·이미지를 반환합니다. 실제
AI 구현에서는 생성 방식만 교체하고 JSON 필드, 상태 코드,
`Idempotency-Key` 계약을 유지합니다.

정답 채점, 재시도·힌트·완료 판정과 단어별 시도 로그 저장은 백엔드가 담당하므로
생성 mock 엔드포인트에 포함하지 않습니다.

## 이야기 생성 연동 계약

오케스트레이션 저장소의 `contracts/openapi/ai-api.yaml`과
`docs/product/features/story-branch.md`를 기준 원본으로 사용합니다.

- 분기 대사의 `content`는 아동에게 표시할 질문입니다.
- `requiresBranchInput=true`이면 `branchPrompt.options`에 서로 다른 선택지 3개를
  제공하며 번호는 정확히 1, 2, 3입니다.
- 일반 대사의 `branchPrompt`는 `null`입니다.
- `continue`의 `branchIntent`는 음성 STT 또는 버튼 선택 결과를 Backend가 확정한
  문자열이며 AI server는 입력 출처를 구분하지 않습니다.
- 교수자 예상 단어는 생성 입력이 아닙니다. 생성된 교안 편집은 Backend가
  `lesson-material` API로 처리합니다.
- 이미지 생성은 현재 동기 응답으로 `imageUrl`만 반환하며 별도 생성 상태값은
  사용하지 않습니다.

실제 생성 provider를 연결할 때는 Mock 생성 함수만 교체하고 요청·응답 모델,
진행률 범위, `requestId`, `schemaVersion`, 서비스 인증과 멱등성 계약을 유지합니다.
