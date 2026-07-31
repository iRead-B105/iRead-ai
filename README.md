# iRead AI

iRead의 FastAPI AI 서버와 개발용 Streamlit 테스트 UI입니다. 핵심 기능은
아동 읽기 프로필에 맞춘 이야기 장 생성, Kiwi·G2P 분석과 후보 선택·국소
교정, 페이지별 `visualScene`, Gemini 삽화 생성입니다.

## 1. 제공 기능

| 구분 | Method | Path | 용도 |
|---|---|---|---|
| 상태 | `GET` | `/health` | 서버·글 공급자 확인 |
| 이야기 | `POST` | `/api/v3/story/chapters/generate` | 실제 개인화 장 생성 |
| 그림 | `POST` | `/api/v1/story/images/generate` | 실제 페이지 삽화 생성 |
| 비교 | `POST` | `/api/dev/story/displayed-chapter-comparison` | 개발용 일반 LLM 비교 |

기존 Backend와의 호환을 위해 훈련·이야기 mock 및 발음 평가 v1 경로도
유지합니다. 실제 개인화 글·그림에는 위 표의 경로를 사용하세요.

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
   ```

   `GMS_KEY`는 AI 서버만 보관하는 환경변수입니다. Swagger·Postman 요청
   헤더에는 넣지 않습니다. 요청의 `X-API-Key`에는
   `AI_INTERNAL_API_KEY` 값을 넣습니다.

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
     "storyProvider": "gms"
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

7. 작업을 마치면 종료합니다.

   ```powershell
   docker compose -f compose.dev.yaml down
   ```

비용 없이 계약만 확인하려면 `.env`의 `STORY_PROVIDER=mock`을 유지하세요.
실제 그림에는 `GMS_KEY`가 필요합니다.

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
