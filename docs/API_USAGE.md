# API 사용 설명서

이 문서는 Docker 개발 환경의 `http://127.0.0.1:8081`을 기준으로 합니다.

## 1. 서버 설정

실제 글과 그림을 생성하려면 `.env`에 다음 값을 설정하고 API를 재시작합니다.

```dotenv
APP_ENV=development
AI_INTERNAL_API_KEY=팀내부통신용긴랜덤문자열
STORY_PROVIDER=gms
OPENAI_MODEL=gpt-5.4-mini
GMS_KEY=발급받은_GMS_키
GMS_GEMINI_IMAGE_MODEL=gemini-2.5-flash-image
```

- `GMS_KEY`: AI 서버가 GMS에 요청할 때만 사용하는 비밀 환경변수
- `AI_INTERNAL_API_KEY`: Backend가 AI 서버에 요청할 때 쓰는 내부 인증값
- `GMS_KEY`를 API Header나 Body에 보내지 않음
- 직접 OpenAI를 호출할 때만 `STORY_PROVIDER=openai`와
  `OPENAI_API_KEY`를 사용

## 2. 공통 Header

이야기 장과 그림 생성 요청에는 다음 세 Header가 필요합니다.

```http
Content-Type: application/json
X-API-Key: <AI_INTERNAL_API_KEY 값>
Idempotency-Key: <1~128자의 요청별 고유값>
```

| Header | 필수 | 설명 |
|---|---|---|
| `Content-Type` | 예 | `application/json` |
| `X-API-Key` | 예 | `.env`의 `AI_INTERNAL_API_KEY`와 정확히 같은 값 |
| `Idempotency-Key` | 예 | 새 요청에는 새 값, 같은 Body 재시도에는 같은 값 |

같은 `Idempotency-Key`와 같은 Body는 이전 응답을 재생합니다. 같은 키에
다른 Body를 보내면 `409 IDEMPOTENCY_CONFLICT`입니다. 추적이 쉽도록
Body의 `requestId`와 같은 값을 쓰는 것을 권장하지만 둘이 반드시 같을
필요는 없습니다.

개발 비교 API는 `Content-Type`, `X-API-Key`만 사용합니다.

## 3. 첫 장 생성

### Endpoint

```http
POST /api/v3/story/chapters/generate
```

첫 장은 `storyRevision: 0`, `chapterNumber: 1`, `branchInput: null`,
빈 `recentPages`, `lastQuestion: null`로 시작합니다. 완전히 복사할 수 있는
Body는
[story-chapter-opening-request.json](examples/story-chapter-opening-request.json)
입니다.

### Body 구조

```json
{
  "requestId": "story-opening-001",
  "schemaVersion": 3,
  "storyId": 101,
  "studentId": 7,
  "storyRevision": 0,
  "chapterNumber": 1,
  "conclude": false,
  "storyTemplate": {
    "templateId": 11,
    "version": 3,
    "title": "토끼와 거북이",
    "context": "빠른 토끼와 느리지만 끈기 있는 거북이가 숲길에서 경주해요.",
    "currentBeat": {
      "beatId": "race-challenge",
      "goal": "토끼가 거북이를 놀리고, 거북이가 경주를 제안해요.",
      "questionFocus": "경주가 시작된 뒤 생길 재미있는 사건",
      "allowedBranchSlots": ["FUNNY_SOUND", "SURPRISING_EVENT"]
    }
  },
  "storyState": {
    "rollingSummary": "",
    "resolvedFacts": [],
    "unresolvedHooks": ["토끼와 거북이가 경주를 시작하려 해요."],
    "recentPages": [],
    "characters": [
      {
        "characterId": "hare",
        "name": "토끼",
        "role": "빠르고 자만하는 주인공",
        "immutableTraits": ["빠르다", "처음에는 거북이를 얕본다"]
      },
      {
        "characterId": "tortoise",
        "name": "거북이",
        "role": "느리지만 끈기 있는 주인공",
        "immutableTraits": ["천천히 간다", "포기하지 않는다"]
      }
    ],
    "lastQuestion": null
  },
  "chapterPlan": {
    "orderedEvents": [
      {
        "eventId": "hare-teases",
        "lockedEvent": "토끼가 느린 거북이를 놀려요.",
        "requiredCharacters": ["hare", "tortoise"],
        "requiredConcepts": ["토끼의 자만", "거북이의 반응"]
      },
      {
        "eventId": "race-begins",
        "lockedEvent": "거북이가 경주를 제안하고 둘이 출발해요.",
        "requiredCharacters": ["hare", "tortoise"],
        "requiredConcepts": ["경주 제안", "출발"]
      }
    ],
    "minPages": 2,
    "maxPages": 4,
    "questionFocus": "경주가 시작된 뒤 생길 재미있는 사건"
  },
  "branchInput": null,
  "generationProfile": {
    "schemaVersion": 2,
    "generationProfileVersion": 7,
    "sourceReadingProfileVersion": 12,
    "compilerVersion": "reading-policy-v2",
    "policyHash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "contentContract": {
      "sentenceCount": 4,
      "preferredWrittenSyllables": {"min": 55, "max": 70},
      "acceptedWrittenSyllables": {"min": 50, "max": 75},
      "directDialogueCount": 1
    },
    "skills": [
      {
        "code": "ONSET_ㄲ",
        "role": "LIMITED",
        "maxOccurrences": 1,
        "targetMin": null,
        "targetMax": null,
        "unitPenalty": 1.2
      },
      {
        "code": "PHONO_LIAISON",
        "role": "EXCLUDED",
        "maxOccurrences": 0,
        "targetMin": null,
        "targetMax": null,
        "unitPenalty": 1.5
      }
    ],
    "protectedTerms": ["토끼", "거북이"]
  }
}
```

### 주요 필드

| 필드 | 규칙 |
|---|---|
| `studentId` | Backend 라우팅용 비식별 ID. 모델에는 전달하지 않음 |
| `storyRevision` | Backend가 소유한 현재 revision |
| `conclude` | 마지막 장이면 `true` |
| `currentBeat` | 전체 동화에서 이번 장이 맡을 목표 |
| `storyState` | 지금까지 확정된 이야기 상태 |
| `chapterPlan.orderedEvents` | 이번 장에서 순서대로 일어날 1~4개 사건 |
| `minPages`, `maxPages` | 각각 2~4, `minPages <= maxPages` |
| `questionFocus` | 비완결 장에는 필수, 완결 장에는 `null` |
| `generationProfile` | Backend가 아동 읽기 프로필을 컴파일한 스냅샷 |
| `protectedTerms` | 고유명사처럼 분석 점수와 별개로 보존할 표현 |

`skills`의 `role`은 `ALLOWED`, `LIMITED`, `EXCLUDED`, `TARGET`입니다.
`LIMITED`·`EXCLUDED`에는 `maxOccurrences`, `TARGET`에는 `targetMin`과
`targetMax`가 필요합니다.

### PowerShell 요청

```powershell
$requestId = "story-opening-001"
$headers = @{
  "X-API-Key" = "팀내부통신용긴랜덤문자열"
  "Idempotency-Key" = $requestId
}

$chapter = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8081/api/v3/story/chapters/generate" `
  -Headers $headers `
  -ContentType "application/json; charset=utf-8" `
  -InFile ".\docs\examples\story-chapter-opening-request.json"
```

응답의 핵심 필드는 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| `pages` | 2~4페이지의 3~4문장, 마지막 질문·선택지, `visualScene` |
| `quality` | 장·페이지별 Kiwi·G2P 읽기 규칙 분석 |
| `generation` | 모델·프롬프트·후보·교정·시각 장면 provenance |
| `timingMs` | 생성·분석·분할·교정·장면 생성 시간 |
| `statePatch` | Backend가 검증 후 기존 상태에 병합할 변경분 |

## 4. 응답 상태 저장

`statePatch`는 완성된 `storyState`가 아니라 변경분입니다. 다음 순서로
Backend 트랜잭션에서 반영합니다.

1. `statePatch.expectedBaseRevision`과 저장된 `storyRevision`이 같은지 확인
2. 이미 반영한 `generationId`인지 확인
3. `rollingSummary` 교체
4. `resolvedFactsAdded`, `unresolvedHooksAdded` 추가
5. `unresolvedHooksRemoved` 제거
6. `charactersUpserted`를 `characterId` 기준으로 병합
7. 응답 페이지를 `recentPages`에 추가하고 최근 8개 이하로 유지
8. `lastQuestion` 교체
9. `storyRevision`을 정확히 1 증가
10. `generationId`를 반영 완료 목록에 저장

`statePatch` 자체를 다음 요청의 `storyState`로 보내면 안 됩니다.
AI 서버의 멱등 저장소는 메모리 기반이므로 영속 상태의 원본은 Backend가
소유합니다.

## 5. 아이 답으로 다음 장 생성

다음 장은 증가한 revision·chapter 번호, 갱신한 `storyState`, 직전 페이지와
질문, 안전 확인과 비식별 처리를 끝낸 `branchInput`을 보냅니다.

완전한 예제:
[story-chapter-continuation-request.json](examples/story-chapter-continuation-request.json)

```json
{
  "storyRevision": 8,
  "chapterNumber": 2,
  "storyState": {
    "rollingSummary": "경주가 시작되었고 토끼가 먼저 달려갔어요.",
    "recentPages": [
      {
        "pageNumber": 3,
        "sentences": [
          "거북이는 높은 언덕 앞에 섰어요.",
          "토끼는 나무 아래에서 쉬었어요.",
          "“나는 계속 갈 거야.” 거북이가 말했어요."
        ],
        "question": "어떤 소리로 경주를 시작할까요?"
      }
    ],
    "lastQuestion": "어떤 소리로 경주를 시작할까요?"
  },
  "branchInput": {
    "source": "TEXT_CONFIRMED",
    "text": "방구 소리로 출발해요!"
  }
}
```

위 블록은 달라지는 부분만 보여 줍니다. 실제 요청에는 첫 장과 같은
`storyTemplate`, `chapterPlan`, 전체 `storyState`,
`generationProfile`도 모두 필요합니다.

`branchInput.source` 허용값은 `CHOICE`, `TEXT_CONFIRMED`,
`STT_CONFIRMED`입니다. 원본 STT, 이름·연락처 등 식별정보, 안전 확인 전
텍스트는 보내지 않습니다.

## 6. 페이지 그림 생성

### Endpoint와 Header

```http
POST /api/v1/story/images/generate
Content-Type: application/json
X-API-Key: <AI_INTERNAL_API_KEY>
Idempotency-Key: <페이지별 고유값>
```

완전한 Body는
[story-image-request.json](examples/story-image-request.json)입니다.

```json
{
  "requestId": "story-image-example-001",
  "schemaVersion": 1,
  "storyId": 101,
  "storyRevision": 8,
  "chapterNumber": 2,
  "pageNumber": 1,
  "sentences": [
    "토끼가 출발선에서 웃어요.",
    "거북이는 종을 바라봐요.",
    "두 친구가 출발을 기다려요."
  ],
  "visualScene": {
    "shot": "WIDE_THREE_QUARTER",
    "characters": [
      {
        "characterId": "hare",
        "present": true,
        "position": "출발선 왼쪽",
        "orientation": "결승선 방향",
        "gazeTarget": "finishBell",
        "action": "두 발을 모으고 출발을 기다림",
        "emotion": {"type": "EXCITED", "intensity": "MEDIUM"}
      },
      {
        "characterId": "tortoise",
        "present": true,
        "position": "출발선 오른쪽",
        "orientation": "결승선 방향",
        "gazeTarget": "finishBell",
        "action": "고개를 들고 차분히 출발을 기다림",
        "emotion": {"type": "FOCUSED", "intensity": "LOW"}
      }
    ],
    "mustInclude": ["출발선", "언덕 너머의 종"],
    "mustAvoid": ["글자", "테두리", "같은 캐릭터 중복"]
  },
  "storyContext": {
    "title": "토끼와 거북이",
    "characters": [
      {
        "characterId": "hare",
        "name": "토끼",
        "role": "주인공",
        "immutableTraits": ["하얀 털", "긴 귀"]
      },
      {
        "characterId": "tortoise",
        "name": "거북이",
        "role": "주인공",
        "immutableTraits": ["초록 등딱지", "차분한 성격"]
      }
    ]
  },
  "characterReferences": []
}
```

장 응답의 해당 페이지 `sentences`와 `visualScene`을 수정하지 않고 그대로
복사합니다. `storyContext.characters`에는 같은 `characterId`를 사용합니다.
승인된 서버 에셋이 없으면 `characterReferences`는 빈 배열입니다.

글을 먼저 보여 주려면 장 응답 직후 모든 페이지 그림을 작업 큐에 넣고 최대
2개씩 병렬 호출합니다. 그림을 장 생성 응답에 동기적으로 묶지 않습니다.

```powershell
$requestId = "story-image-example-001"
$headers = @{
  "X-API-Key" = "팀내부통신용긴랜덤문자열"
  "Idempotency-Key" = $requestId
}

$image = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8081/api/v1/story/images/generate" `
  -Headers $headers `
  -ContentType "application/json; charset=utf-8" `
  -InFile ".\docs\examples\story-image-request.json"

[IO.File]::WriteAllBytes(
  ".\page-1.png",
  [Convert]::FromBase64String($image.imageBase64)
)
```

응답은 `mimeType`, `imageBase64`, `model`, `promptVersion`,
`timingMs`를 반환합니다.

## 7. Swagger에서 테스트

1. `http://127.0.0.1:8081/docs` 접속
2. 오른쪽 위 **Authorize** 클릭
3. `AI_INTERNAL_API_KEY` 원문만 입력하고 Authorize
4. 원하는 API 펼치기
5. **Try it out** 클릭
6. `Idempotency-Key`에 요청별 고유값 입력
7. Request body에 완전한 JSON 붙여넣기
8. **Execute** 클릭

`Bearer ` 접두어와 `GMS_KEY`를 입력하지 않습니다.

## 8. Postman에서 테스트

1. Method를 `POST`로 선택
2. URL 입력
3. Authorization 탭은 `No Auth`
4. Headers에 다음 행 추가

   | Key | Value |
   |---|---|
   | `Content-Type` | `application/json` |
   | `X-API-Key` | `.env`의 `AI_INTERNAL_API_KEY` |
   | `Idempotency-Key` | 예: `story-opening-001` |

5. Body → raw → JSON 선택
6. `docs/examples`의 완전한 JSON 붙여넣기
7. **Send**

## 9. 개발 비교 API

```http
POST /api/dev/story/displayed-chapter-comparison
Content-Type: application/json
X-API-Key: <AI_INTERNAL_API_KEY>
```

Body:

```json
{
  "requestId": "comparison-001",
  "chapterRequest": {
    "...": "실제 v3 장 요청 전체"
  },
  "personalizedResponse": {
    "...": "그 요청으로 화면에 표시한 v3 응답 전체"
  }
}
```

동일 맥락의 일반 LLM 결과를 한 번 더 생성하므로 시간과 비용이 추가됩니다.
`APP_ENV=production`에서는 404입니다.

## 10. 오류와 재시도

공통 오류 형태:

```json
{
  "code": "INVALID_API_KEY",
  "message": "내부 API 키 인증에 실패했습니다.",
  "requestId": "story-opening-001",
  "retryable": false
}
```

| HTTP | 대표 코드 | 조치 |
|---:|---|---|
| 400 | `MISSING_IDEMPOTENCY_KEY` | Header 추가 |
| 400 | `INVALID_IDEMPOTENCY_KEY` | 1~128자로 수정 |
| 400 | `INVALID_JSON` | JSON 문법 수정 |
| 400 | `INVALID_REQUEST_FORMAT` | 필드·타입·revision 확인 |
| 401 | `INVALID_API_KEY` | 내부 키 일치 확인 |
| 409 | `IDEMPOTENCY_CONFLICT` | 변경된 Body에는 새 키 사용 |
| 409 | `IDEMPOTENCY_IN_PROGRESS` | 잠시 후 같은 키로 확인 |
| 422 | 상태·정책·레퍼런스 오류 | 메시지에 나온 계약 수정 |
| 500 | `INTERNAL_RESPONSE_ERROR` | 서버 로그 확인 |
| 502 | 모델 응답 오류 | 같은 키로 최대 1회 재시도 |
| 503 | `IMAGE_PROVIDER_NOT_CONFIGURED` | `GMS_KEY` 설정 |
| 504 | 공급자 시간 초과 | 같은 키로 최대 1회 재시도 |

연결 실패와 502·503·504만 동일한 Body와 멱등키로 최대 1회 재시도합니다.
키, 원본 아동 식별정보, 원본 STT, 모델 원문은 로그에 남기지 않습니다.

## 11. 호환 API

아래 경로는 기존 Backend 계약을 깨지 않기 위해 유지합니다.

- `POST /api/v1/trainings/candidates`
- `POST /api/v1/trainings/generate`
- `POST /api/v1/story/generate`
- `POST /api/v1/story/continue`
- `POST /api/v1/images/generate`
- `GET /api/v1/images/mock/generated.svg`
- `POST /api/v1/speech/pronunciation/analyze`

`/api/v1/story/generate`, `/continue`, `/api/v1/images/generate`는 실제
개인화 LLM·Gemini API가 아니라 호환용 결정적 mock입니다. 호환 API의
인증·멱등 계약은 Swagger의 각 경로를 따릅니다.

## 12. 프롬프트 수정

프롬프트는 `src/iread_ai/prompts/`의 Markdown 파일입니다.

| 파일 | 역할 |
|---|---|
| `chapter_personalized.md` | 개인화 장 후보 |
| `chapter_baseline.md` | 일반 비교 장 |
| `page_repair.md` | 최대 2문장 국소 교정 |
| `visual_scene.md` | 구조화된 페이지 장면 |
| `story_image.md` | Gemini 삽화 |

수정 후 `uv run pytest`를 실행합니다. 응답의 `promptVersion`,
`visualScenePromptVersion`, 이미지 `promptVersion`으로 어떤 버전을
사용했는지 확인할 수 있습니다.
