# iRead AI

iRead의 FastAPI AI 서버와 개발용 Streamlit 테스트 UI입니다. 핵심 기능은
아동 읽기 프로필에 맞춘 이야기 장 생성, Kiwi·G2P 분석과 후보 선택·국소
교정, 페이지별 `visualScene`, Gemini 삽화 생성입니다.

## 1. 제공 기능

| 구분 | Method | Path | 용도 |
|---|---|---|---|
| 상태 | `GET` | `/health` | 서버·글 공급자 확인 |
| 이야기 나라 시작 | `POST` | `/api/v1/story/generate` | 기존 Backend 계약으로 실제 모델 기반 글 생성 |
| 이야기 나라 이어쓰기 | `POST` | `/api/v1/story/continue` | 아이의 분기를 반영한 실제 모델 기반 글 생성 |
| 이야기 | `POST` | `/api/v3/story/chapters/generate` | 실제 개인화 장 생성 |
| 그림 | `POST` | `/api/v1/story/images/generate` | 실제 페이지 삽화 생성 |
| 비교 | `POST` | `/api/dev/story/displayed-chapter-comparison` | 개발용 일반 LLM 비교 |

기존 Backend의 이야기 나라가 사용하는 v1 이야기 경로는 v3 개인화 생성기를
호출한 뒤 기존 5줄 응답으로 변환합니다. 후보 3개를 Kiwi·G2P로 비교하고,
필요한 문장만 조건부 LLM 국소 교정한 뒤 점수가 좋아질 때만 채택합니다.
`STORY_PROVIDER=mock`이면 결정적 mock을, `gms` 또는 `openai`이면 실제 모델을
사용합니다. 훈련 및 발음 평가 호환 경로도 그대로 유지합니다.

## 2. Docker로 처음 실행

필요한 것은 Docker Desktop과 GMS 키입니다.

1. 환경 파일을 만듭니다.

   ```powershell
   Copy-Item .env.example .env
   ```

2. `.env`에서 다음 세 값을 설정합니다.

   ```dotenv
   APP_ENV=development
   AI_INTERNAL_API_KEY=팀내부통신용긴랜덤문자열
   STORY_PROVIDER=gms
   GMS_KEY=발급받은_GMS_키
   OPENAI_MODEL=gpt-5.4-mini
   STORY_IMAGE_PROVIDER=disabled
   ```

   `GMS_KEY`는 AI 서버만 보관하는 환경변수입니다. Swagger·Postman 요청
   헤더에는 넣지 않습니다. 요청의 `X-API-Key`에는
   `AI_INTERNAL_API_KEY` 값을 넣습니다. `OPENAI_MODEL`은 GMS의 OpenAI
   호환 API를 사용할 때도 적용되는 글 모델 설정입니다.

3. API와 UI를 빌드하고 실행합니다.

   ```powershell
   docker compose -f compose.dev.yaml up -d --build
   ```

4. 컨테이너와 상태를 확인합니다.

   ```powershell
   docker compose -f compose.dev.yaml ps
   Invoke-RestMethod http://127.0.0.1:8081/health
   ```

   정상 응답:

   ```json
   {
     "status": "UP",
     "service": "iread-ai",
     "storyProvider": "gms",
     "storyImageProvider": "disabled"
   }
   ```

5. 브라우저로 접속합니다.

   - Swagger: `http://127.0.0.1:8081/docs`
   - 개발 테스트 UI: `http://127.0.0.1:8506`

6. 로그를 확인합니다.

   ```powershell
   docker compose -f compose.dev.yaml logs -f --tail=100 api
   docker compose -f compose.dev.yaml logs -f --tail=100 ui
   ```

   이야기 요청 상태와 개인화 품질만 필터링하려면 다음 명령을 사용합니다.

   ```powershell
   docker compose -f compose.dev.yaml logs -f --tail=100 api 2>&1 |
     Select-String '"event":"story_generation_'
   ```

7. 작업을 마치면 종료합니다.

   ```powershell
   docker compose -f compose.dev.yaml down
   ```

### 기존 Backend 이야기 나라에 연결

Backend와 AI 서버가 함께 실행될 때 Backend에 다음 환경변수를 설정합니다.

```dotenv
AI_MOCK_GENERATE=false
# 현재 전체 데모 Compose에서 AI 컨테이너를 교체한 경우
AI_BASE_URL=http://iread-ai-mock:8080
AI_API_KEY=AI서버의_AI_INTERNAL_API_KEY와_같은_값
```

Backend를 호스트에서 실행하면 `AI_BASE_URL=http://127.0.0.1:8081`, Docker
Backend가 호스트의 AI 서버를 호출하면
`AI_BASE_URL=http://host.docker.internal:8081`을 사용합니다.

AI 서버에는 `STORY_PROVIDER=gms`와 `GMS_KEY`를 설정합니다. 이미지 비용 없이
글 연결만 확인할 때는 `STORY_IMAGE_PROVIDER=disabled`를 유지합니다. 이 경우
기존 Backend가 호출하는 `/api/v1/images/generate`는 짧은 mock SVG URL만
반환하며 Gemini 이미지 생성은 호출하지 않습니다.

현재 Backend v1 요청에는 읽기 프로필 스냅샷이 없으므로 호환 어댑터는
임시 균형형 프로필을 사용합니다. 실제 아동별 개인화에는 Backend가 v3의
`generationProfile`을 전달하는 후속 연동이 필요합니다.

비용 없이 계약만 확인하려면 `.env`의 `STORY_PROVIDER=mock`을 유지하세요.
그림 생성은 기본적으로 꺼져 있습니다. 실제 그림을 테스트할 때만
`STORY_IMAGE_PROVIDER=gemini`와 `GMS_KEY`를 함께 설정하세요.

## 3. 요청해 보기

첫 장 요청 예제를 그대로 보냅니다.

```powershell
$requestId = "story-opening-001"
$headers = @{
  "X-API-Key" = "팀내부통신용긴랜덤문자열"
  "Idempotency-Key" = $requestId
}

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8081/api/v3/story/chapters/generate" `
  -Headers $headers `
  -ContentType "application/json; charset=utf-8" `
  -InFile ".\docs\examples\story-chapter-opening-request.json"
```

정확한 Header, 첫 장·다음 장 Body, 상태 반영, 그림 요청, Swagger와
Postman 클릭 순서는 [API 사용 설명서](docs/API_USAGE.md)에 있습니다.

## 4. Docker 없이 개발

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
Copy-Item .env.example .env
uv sync --extra dev --extra ui
uv run uvicorn iread_ai.app:app --host 0.0.0.0 --port 8081
```

다른 PowerShell에서 UI를 실행합니다.

```powershell
uv run streamlit run streamlit_app.py --server.port 8506
```

## 5. 검증

```powershell
uv run ruff check .
uv run pytest
docker compose -f compose.dev.yaml config
```

## 6. 문서

- [API 사용 설명서](docs/API_USAGE.md)
- [테스트 UI 사용 설명서](docs/TEST_UI.md)
- [개인화 비교 근거](docs/PERSONALIZATION_BENCHMARK.md)
- [Backend·Orchestration 통합 경계](docs/INTEGRATION_BOUNDARY.md)
- [선택적 캐릭터 에셋](assets/README.md)

## 7. 프롬프트 수정

긴 프롬프트는 `src/iread_ai/prompts/`의 Markdown 파일로 분리했습니다.

| 파일 | 역할 |
|---|---|
| `chapter_personalized.md` | 아동 프로필 기반 장 후보 생성 |
| `chapter_baseline.md` | 개발 비교용 일반 장 생성 |
| `page_repair.md` | 선택 결과의 최대 2문장 국소 교정 |
| `visual_scene.md` | 페이지별 구조화 장면 생성 |
| `story_image.md` | Gemini 삽화 지시 |

수정 후 테스트를 실행하세요. 응답의 프롬프트 버전에는 파일 해시가 포함되어
결과를 역추적할 수 있습니다.

저장소 병합과 Orchestration submodule 포인터 갱신 순서는
[통합 경계 문서](docs/INTEGRATION_BOUNDARY.md)에 정리했습니다.

## 8. 이야기 생성 연동 계약

정답 채점, 재시도·힌트·완료 판정과 단어별 시도 로그 저장은 백엔드가 담당하므로
생성 mock 엔드포인트에 포함하지 않습니다.

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
