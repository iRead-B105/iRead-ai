# iRead AI

iRead의 FastAPI 기반 AI 서비스입니다. Azure 단어별 발음 평가와 실제 AI 서버
구현팀이 참고할 수 있는 결정적 생성 mock API를 함께 제공합니다.

## 제공 엔드포인트

| Method | Path | 역할 |
|---|---|---|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/api/v1/trainings/candidates` | 34개 훈련 타입별 문항 후보 생성 |
| `POST` | `/api/v1/trainings/generate` | 레거시 훈련 데이터 envelope 생성 |
| `POST` | `/api/v1/story/generate` | 최초 이야기 대사 1~5 생성 |
| `POST` | `/api/v1/story/continue` | 분기 선택을 반영한 대사 6~10 생성 |
| `POST` | `/api/v1/images/generate` | 훈련 장면·이야기 친구 이미지 URL 생성 |
| `GET` | `/api/v1/images/mock/generated.svg` | 생성된 mock SVG 조회 |
| `POST` | `/api/v1/speech/pronunciation/analyze` | Azure 단어별 발음 평가 |

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
