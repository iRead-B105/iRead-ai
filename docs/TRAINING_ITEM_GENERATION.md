# 훈련 문항 후보 생성과 검토

## 권장 방식: 맞춤 훈련 5종 세트

실서비스에서는 같은 글자를 같은 형식으로 5번 반복하지 않는다. 하나의 핵심 읽기
목표를 서로 다른 활동 5개로 연습한다. 예를 들어
`GRAPHEME.VOWEL.BASIC.ㅏ`를 요청하면 다음과 같이 구성한다.

1. 모음 `ㅏ` 따라 보기
2. `ㅏ` 소리 듣고 고르기
3. 자음과 모음 구별하기
4. `ㄱ + ㅏ`처럼 음소를 합쳐 음절 만들기
5. `가`처럼 기본 글자 만들기

`soundText`는 후보를 구분하기 위한 문구가 아니다. 화면의 `target`을 TTS로 들려줄
때 사용하는 표준 발음이다. 따라서 `ㅏ`의 `soundText`는 항상 `아`이며 규칙 사전에
저장된 값만 사용한다. `traceAssetKey`도 프론트에 실제로 존재하는 획순 에셋만 사용한다.

새 세트 생성 API:

```http
POST /api/v1/training-sets/generate
```

```json
{
  "requestId": "training-set-001",
  "schemaVersion": 1,
  "curriculumArea": "LETTER_SOUND",
  "activityCount": 5,
  "difficulty": 1,
  "targetFeatures": [
    {
      "featureCode": "GRAPHEME.VOWEL.BASIC.ㅏ",
      "weaknessScore": 0.8,
      "confidence": 0.9,
      "evidenceCount": 10
    }
  ],
  "excludedFeatures": [],
  "preferredTrainingTypes": [],
  "additionalPrompt": ""
}
```

활동 하나만 다시 생성하는 API:

```http
POST /api/v1/training-activities/generate
```

```json
{
  "requestId": "training-activity-001",
  "schemaVersion": 1,
  "sequence": 2,
  "trainingType": "VOWEL_SOUND_CHOICE",
  "difficulty": 1,
  "targetFeatures": [
    {
      "featureCode": "GRAPHEME.VOWEL.BASIC.ㅏ",
      "weaknessScore": 0.8,
      "confidence": 0.9,
      "evidenceCount": 10
    }
  ],
  "excludedFeatures": [],
  "additionalPrompt": ""
}
```

검토 UI는 `training_set_review_app.py`이며 각 활동을 따로 채택하거나 다시 생성할 수
있다. 기존 `/api/v1/trainings/candidates`와 `training_review_app.py`는 현재 백엔드 계약
호환과 34종 원본 형식 진단을 위해 유지한다.

## 생성 전략

`POST /api/v1/trainings/candidates`는 백엔드가 전달한 훈련 유형, 난이도, 목표 특성,
제외 특성을 이용해 서로 다른 후보 5개를 반환한다.

- `rule-db`: 정답을 기계적으로 검증할 수 있는 24개 유형이다. AI 서버의 SQLite 문항
  원자 사전에서 아동 프로필에 맞는 글자·음절·단어를 고르고 정답과 오답을 조합한다.
- `gms:{model}`: 자연스러운 문장이나 짧은 글이 필요한 10개 유형이다. GMS LLM이
  생성한 JSON을 Pydantic 계약, 유형별 의미 규칙, 아동 안전 규칙으로 검사한다.
- `curated-fallback`: `AI_GENERATION_PROVIDER=mock` 모드에서 반환하는 검증용 고정
  후보이다. LLM이 생성한 결과라는 뜻이 아니다. 실제 provider(GMS·OpenAI·Gemini)가
  생성·검사에 실패하면 고정 후보로 대체하지 않고 502(재시도 불가) 또는
  503(재시도 가능) 오류를 반환한다.

규칙·사전 방식은 다음 24개 유형에 적용된다.

```text
VOWEL_TRACE, CONSONANT_TRACE, SYLLABLE_TRACE,
CONSONANT_SOUND_CHOICE, VOWEL_SOUND_CHOICE,
CONSONANT_VOWEL_CLASSIFICATION,
SYLLABLE_INITIAL_CHOICE, WORD_INITIAL_CHOICE,
SAME_INITIAL_WORD_CHOICE,
FINAL_CONSONANT_CHOICE, WORD_FINAL_SOUND_CHOICE,
FINAL_CONSONANT_COMPARISON, SIMILAR_SOUND_CHOICE,
PHONEME_BLEND, SYLLABLE_BLEND,
BASIC_SYLLABLE_BUILD, FINAL_SYLLABLE_BUILD, DOUBLE_FINAL_BUILD,
FINAL_CONSONANT_DELETE, SYLLABLE_DELETE, SYLLABLE_REPLACE,
WORD_READING, NONWORD_READING, WORD_CHAIN_READING
```

LLM과 로컬 검사를 함께 사용하는 유형은 다음 10개이다.

```text
DIFFICULT_WORD_PREVIEW, SENTENCE_READING, SHORT_PASSAGE_READING,
SENTENCE_ASSEMBLY, FILL_IN_THE_BLANK, IMAGE_SENTENCE_MATCH,
SENTENCE_REPEAT, PHRASE_READING, REPEATED_SENTENCE_READING,
SHORT_STORY_READING
```

어휘 검증은 훈련 단계에 따라 다르게 적용한다.

- 글자·음절·낱말 단위 24종은 `PSEUDOWORD_ALLOWED`이다. 난독 훈련에 필요한
  무의미 글자·음절·낱말을 허용하며 사전 미등재만으로 탈락시키지 않는다.
- 문장·짧은 글 단위 10종은 `REAL_WORD_ONLY`이다. Kiwi가 내용어의 기본형을
  분석하고 AI 서버 소유 사전에서 등재 여부를 확인한다. `까나`처럼 목표 글자를
  맞추기 위해 만든 조어가 있으면 후보 전체를 다시 생성하고, 재시도도 실패하면
  503 오류를 반환한다.
- 고유명사는 Kiwi의 고유명사 판정을 따르므로 일반 등재어 검사에서 제외한다.

## API 실행

AI 서버 저장소에서 환경 파일을 준비한다.

```powershell
Copy-Item .env.example .env
```

비용 없이 전체 계약과 UI를 확인할 때는 다음 설정을 사용한다.

```dotenv
AI_GENERATION_PROVIDER=mock
STORY_IMAGE_PROVIDER=disabled
```

문장형 10개 유형까지 GMS로 생성할 때는 다음 설정을 사용한다.

```dotenv
AI_GENERATION_PROVIDER=gms
GMS_KEY=발급받은_GMS_키
OPENAI_MODEL=gpt-5.4-mini
STORY_IMAGE_PROVIDER=disabled
```

OpenAI API를 직접 사용할 때는 다음 설정을 사용한다.

```dotenv
AI_GENERATION_PROVIDER=openai
OPENAI_API_KEY=발급받은_OpenAI_API_키
OPENAI_MODEL=gpt-5.4-mini
STORY_IMAGE_PROVIDER=disabled
```

Docker Desktop이 실행 중인 상태에서 통합 Compose의 AI 서비스만 다시 빌드한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test"
docker compose up -d --build ai
Invoke-RestMethod http://127.0.0.1:8081/health
```

Docker를 사용하지 않을 때는 다음처럼 실행한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"
uv sync --extra dev --extra ui
uv run uvicorn iread_ai.app:app --host 127.0.0.1 --port 8081
```

## 검토 UI 실행

API가 `8081`에서 실행 중일 때 별도 터미널에서 실행한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"
uv run streamlit run training_set_review_app.py --server.port 8507
```

또는 UI만 Docker로 실행한다.

```powershell
docker compose -f compose.training-ui.yaml up -d --build
```

브라우저에서 `http://127.0.0.1:8507`을 연 뒤 다음 순서로 검토한다.

1. 잘 읽는 아이, 조금 어려워하는 아이, 많이 어려워하는 아이 중 하나를 선택한다.
2. 훈련 영역과 오늘의 핵심 목표 특성을 선택한다.
3. `맞춤 훈련 5종 생성`을 눌러 서로 다른 활동 구성을 확인한다.
4. 만족하는 활동은 `이 활동 채택`을 누른다.
5. 만족하지 않는 활동만 `이 활동 다시 생성`으로 교체한다.
6. 5개를 모두 채택하면 검토 결과를 JSON으로 내려받는다.

화면의 `생성 방식`은 실제 응답 헤더 `X-AI-Provider`를 보여 준다. 따라서
`curated-fallback`을 실제 LLM 생성 결과로 오해하지 않고 구분해서 평가할 수 있다.

## Postman 요청

```http
POST http://127.0.0.1:8081/api/v1/trainings/candidates
Content-Type: application/json
X-API-Key: .env의 AI_INTERNAL_API_KEY
Idempotency-Key: training-review-001
```

```json
{
  "requestId": "training-review-001",
  "schemaVersion": 2,
  "trainingType": "CONSONANT_SOUND_CHOICE",
  "count": 5,
  "difficulty": 2,
  "targetFeatures": [
    {
      "featureCode": "GRAPHEME.ONSET.BASIC.ㅅ",
      "weaknessScore": 0.8,
      "confidence": 0.9,
      "evidenceCount": 10
    }
  ],
  "excludedFeatures": [],
  "additionalPrompt": "",
  "outputTemplate": {
    "type": "CONSONANT_SOUND_CHOICE",
    "data": [
      {
        "audioText": "<string>",
        "choices": ["<string>"],
        "answerIndex": "<integer>"
      }
    ]
  }
}
```

응답 본문은 후보 5개이며 실제 생성 방식은 응답 헤더와
`generationMetadata`에서 함께 확인한다.

```json
{
  "type": "CONSONANT_SOUND_CHOICE",
  "data": [
    {
      "audioText": "ㅅ",
      "choices": ["ㄱ", "ㄴ", "ㅅ"],
      "answerIndex": 2
    }
  ],
  "generationMetadata": {
    "provider": "rule-db",
    "model": "korean-training-bank-v1",
    "strategy": "RULE_DB",
    "lexicalPolicy": "PSEUDOWORD_ALLOWED",
    "lexiconApplied": true
  }
}
```

문장형 OpenAI/GMS 응답은 실제 경로에 따라 `provider: "openai"` 또는
`provider: "gms"`, `strategy: "LLM_WITH_LOCAL_VALIDATION"`,
`lexicalPolicy: "REAL_WORD_ONLY"`로 표시된다.
백엔드는 이 객체를 교안의 `generationMetadata`에 보존하므로 교수자 화면이나
저장 데이터에서도 실제 생성 경로를 `MOCK`과 구분할 수 있다.

## 검증

```powershell
uv run ruff check .
uv run pytest
```

테스트는 24개 규칙 유형 각각에 대해 후보 5개의 중복 여부와 함께 초성, 받침,
음소·음절 조합, 삭제·대치 결과, 정답 인덱스의 의미 일치까지 검사한다.
