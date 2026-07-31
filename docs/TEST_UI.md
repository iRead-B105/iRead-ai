# 개발 테스트 UI 사용 설명서

`streamlit_app.py`는 v3 이야기·그림 API를 확인하는 개발 도구입니다.
제품 Frontend가 아니며 아동에게 직접 노출하는 운영 UI가 아닙니다.

## 1. Docker 실행

1. 환경 파일을 준비합니다.

   ```powershell
   Copy-Item .env.example .env
   ```

2. 실제 생성 테스트라면 `.env`를 설정합니다.

   ```dotenv
   AI_INTERNAL_API_KEY=팀내부통신용긴랜덤문자열
   STORY_PROVIDER=gms
   GMS_KEY=발급받은_GMS_키
   ```

3. API와 UI를 실행합니다.

   ```powershell
   docker compose -f compose.dev.yaml up -d --build
   docker compose -f compose.dev.yaml ps
   ```

4. `http://127.0.0.1:8506`을 엽니다.

Compose 내부에서 UI는 `api:8080`으로 요청합니다. 호스트에서 API는
`http://127.0.0.1:8081`입니다.

## 2. 화면 확인 순서

1. 준비된 이야기 10개 중 하나 선택
2. `잘 읽는 아이`, `조금 어려워하는 아이`, `많이 어려워하는 아이` 중 선택
3. 첫 장 생성
4. 페이지 글, `visualScene`, 품질 점수와 단계별 시간 확인
5. 다음 페이지로 이동하면서 완료된 그림이 자동 반영되는지 확인
6. 마지막 페이지 질문에 선택지 또는 자유 답 입력
7. 다음 장을 생성하고 아이 답이 한 페이지 동안 반영된 뒤 본 흐름으로
   돌아오는지 확인
8. 개발 비교 패널에서 화면에 표시된 개인화 결과와 일반 LLM 결과 비교

모든 페이지 글은 장 응답과 함께 먼저 표시됩니다. 그림은 응답 직후 작업
대기열에 들어가 최대 2개씩 병렬 생성되며, 화면을 이동하지 않아도 완료된
그림이 채워집니다. 현재 장의 그림은 최대 4개만 세션에 보관합니다.

대략적인 UI 제한 시간은 글 50초, 그림 200초입니다. 공급자 상태와 페이지
수에 따라 실제 시간은 달라집니다.

비교 버튼은 자동 호출되지 않습니다. 누르면 일반 장 생성을 한 번 더 요청해
비용과 대기 시간이 추가됩니다.

## 3. 화면 지표

- `quality`: Kiwi·G2P 분석 결과와 프로필별 초과 횟수
- `generation`: 후보 수, 선택 후보, 교정 시도·채택 여부
- `timingMs`: 글 생성, 분석, 페이지 분할, 교정, `visualScene`, 총 시간
- 비교 점수: 동일 프로필의 확인 가능한 규칙으로 개인화·일반 결과 비교

비교 점수는 개발 지표이며 치료·진단 결과가 아닙니다. G2P가 일부 규칙을
확정하지 못하면 검증 가능한 표면 규칙만으로 참고 점수를 표시합니다.

## 4. Docker 없이 실행

첫 번째 PowerShell:

```powershell
Copy-Item .env.example .env
uv sync --extra dev --extra ui
uv run uvicorn iread_ai.app:app --host 0.0.0.0 --port 8081
```

두 번째 PowerShell:

```powershell
uv run streamlit run streamlit_app.py --server.port 8506
```

## 5. 문제 해결

### 글 생성 모델에 연결하지 못함

```powershell
Invoke-RestMethod http://127.0.0.1:8081/health
docker compose -f compose.dev.yaml logs --tail=200 api
```

실제 글은 `storyProvider`가 `gms` 또는 `openai`여야 합니다. `.env`를
바꿨다면 컨테이너를 다시 만듭니다.

```powershell
docker compose -f compose.dev.yaml up -d --build --force-recreate
```

### 그림 공급자가 설정되지 않음

`.env`에 유효한 `GMS_KEY`를 넣고 API를 다시 시작합니다.
`STORY_PROVIDER=mock`이어도 `GMS_KEY`가 있으면 글은 mock, 그림은 Gemini로
따로 확인할 수 있습니다.

### `401 INVALID_API_KEY`

`.env`의 `AI_INTERNAL_API_KEY`와 UI 컨테이너에 전달된 값이 같은지 확인합니다.

```powershell
docker compose -f compose.dev.yaml config
```

### `409 IDEMPOTENCY_CONFLICT`

요청 Body를 바꿨다면 새 `requestId`와 새 `Idempotency-Key`를 사용합니다.

### `8081` 또는 `8506` 포트 사용 중

기존 로컬 Python·Streamlit 프로세스나 컨테이너를 종료한 뒤 다시 실행합니다.

```powershell
Get-NetTCPConnection -LocalPort 8081,8506 -State Listen
docker ps
```

### UI 로그

```powershell
docker compose -f compose.dev.yaml logs --tail=200 ui
```
