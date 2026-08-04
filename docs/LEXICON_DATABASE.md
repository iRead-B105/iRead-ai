# 어휘·발음·음운 특징 DB

## 목적과 소유권

이 DB는 AI 서버가 소유합니다. Backend는 학생 식별정보나 원시 학습 이력을 DB에 복제하지 않고, 학생 프로필에서 계산한 읽기 정책만 AI 서버에 전달합니다.

AI 서버는 다음 데이터를 관리합니다.

- 표제어, 품사, 뜻풀이, 어휘 등급
- 표제어와 활용형의 표기 및 사전 발음
- 음절별 초성·중성·종성
- 받침 수와 받침 비율
- 연음, 비음화, 유음화, 구개음화, 된소리되기 등 음운 특징
- 동화 적합도와 동물·자연·행동 등의 의미 태그
- 분석 신뢰도와 사람 검토 필요 여부

학생별 숙달도, 오답 이력, 커리큘럼, 교안 결과는 Backend 소유 데이터입니다.

## 현재 데이터

한국어기초사전 2026-07 내보내기 파일을 바탕으로 구축한 SQLite 스키마 v1을 사용합니다.

| 항목 | 수량 |
|---|---:|
| 표제어 | 56,555 |
| 동화 핵심 어휘 | 3,141 |
| 표기·활용형 | 103,848 |
| 특징 레코드 | 1,148,848 |
| 발음 검토 필요 | 12,461 |

원문 텍스트 라이선스는 `CC BY-SA 2.0 KR`이며 멀티미디어는 포함하지 않습니다. 생성된 SQLite 파일은 100MB 이상이므로 Git에 올리지 않고 배포 볼륨이나 오브젝트 스토리지로 전달합니다.

## 테이블

```text
source_releases 1 ── N lexemes 1 ── N word_forms 1 ── N form_features
```

- `source_releases`: 원본 버전, 해시, 라이선스, 적재 시각
- `lexemes`: 표제어, 품사, 뜻풀이, 어휘 등급, 의미 태그, 동화 적합도
- `word_forms`: 실제 표기, 발음, 음절 수, 받침 수·비율, 발음 분석 상태
- `form_features`: 초중종성과 음운 특징, 발생 횟수, 신뢰도, 분석기 버전
- `metadata`: DB·원본·분석기 버전과 집계값

DB 내부의 `ONSET_ㄲ`, `NUCLEUS_ㅏ`, `CODA_ㄴ`은 API에서 각각 `GRAPHEME.ONSET.TENSE.ㄲ`, `GRAPHEME.VOWEL.BASIC.ㅏ`, `GRAPHEME.CODA.SIMPLE.ㄴ`으로 변환됩니다. 프로젝트 공용 코드가 더 구체적인 경우 부모 코드가 자식 코드를 만족한 것으로 잘못 판정하지 않습니다.

## 로컬 설치

AI 저장소에서 실행합니다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"

& ".\.venv\Scripts\python.exe" -m iread_ai.lexicon.install `
  --source-db "C:\path\to\story-lexicon-2026-07.sqlite3" `
  --destination ".\local-output\lexicon\story-lexicon.sqlite3"

$env:AI_LEXICON_DB_PATH = ".\local-output\lexicon\story-lexicon.sqlite3"
& ".\.venv\Scripts\uvicorn.exe" iread_ai.app:app --host 127.0.0.1 --port 8081
```

설치기는 스키마 버전, 필수 테이블, SQLite 무결성, 원본과 대상의 레코드 수를 확인한 뒤 원자적으로 교체합니다.

Docker 운영에서는 DB 파일을 이미지에 넣지 않고 읽기 전용 볼륨으로 마운트합니다.

```yaml
environment:
  AI_LEXICON_DB_PATH: /data/lexicon/story-lexicon.sqlite3
volumes:
  - ./runtime-data/lexicon:/data/lexicon:ro
```

## 상태 확인 API

```http
GET /api/v1/lexicon/status
X-API-Key: <AI_INTERNAL_API_KEY>
```

정상 응답은 `status=READY`, DB·분석기 버전과 각 레코드 수를 반환합니다. DB가 없거나 손상돼도 서버 시작은 유지하며 `status=UNAVAILABLE`로 원인을 표시합니다.

## 어휘 팔레트 조회 API

```http
POST /api/v1/lexicon/palettes/query
Content-Type: application/json
X-API-Key: <AI_INTERNAL_API_KEY>
```

```json
{
  "requestId": "palette-kk-001",
  "schemaVersion": 1,
  "targetFeatures": [
    {
      "featureCode": "GRAPHEME.ONSET.TENSE.ㄲ",
      "weaknessScore": 0.82,
      "confidence": 0.9
    }
  ],
  "excludedFeatures": [
    "PHONO_LIAISON",
    "GRAPHEME.CODA.COMPLEX"
  ],
  "masteredFeatures": [
    "GRAPHEME.ONSET.BASIC",
    "GRAPHEME.VOWEL.BASIC"
  ],
  "semanticTags": ["ANIMAL", "NATURE"],
  "partsOfSpeech": ["명사", "동사"],
  "minSyllables": 1,
  "maxSyllables": 4,
  "maxBatchimRatio": 0.5,
  "strictPronunciation": true,
  "requireTarget": false,
  "includeInflections": false,
  "limit": 30
}
```

`excludedFeatures`와 받침 비율은 SQL에서 먼저 제거하는 하드 제약입니다. `targetFeatures`는 기본적으로 추천 점수에 반영하고, 목표 특징이 반드시 들어간 연습어만 필요할 때 `requireTarget=true`를 사용합니다. `masteredFeatures` 밖의 글자 특징은 감점하므로 목표 단어에 아직 어려운 특징이 과도하게 끼어드는 것을 줄입니다.

응답에는 각 단어의 표기, 발음, 품사, 뜻풀이, 모든 특징, 점수와 선정 이유가 포함됩니다. LLM에는 전체 사전을 전달하지 않고 이 API가 고른 20~40개 팔레트만 전달합니다.

## 교안 생성에 적용

`/api/v1/training-sets/generate`와 `/api/v1/training-activities/generate` 요청의
`useLexicon` 기본값은 `true`입니다. 문장·구절·짧은 글처럼 LLM을 사용하는 훈련은
다음 순서로 생성합니다.

1. `targetFeatures`, `excludedFeatures`, 난이도로 어휘 팔레트를 조회합니다.
2. 검증된 추천 어휘를 GMS 프롬프트에 전달해 후보 5개를 생성합니다.
3. 각 후보를 Kiwi·G2P와 한글 자모 분석기로 검사합니다.
4. 목표 특징 출현은 가점하고 회피 특징 출현은 감점합니다.
5. 가장 높은 점수의 후보와 선택 근거를 반환합니다.

응답의 `activity.personalization`에서 추천 어휘, 후보별 점수, 목표·회피 특징
발생 수, 팔레트 어휘 사용 여부와 최종 선택 후보를 확인할 수 있습니다. DB가 없으면
서비스를 중단하지 않고 추천 어휘 없이 후보 생성·분석을 계속합니다.

비교를 위해 `useLexicon=false`로 같은 요청을 보내면 어휘 팔레트 주입만 끌 수
있습니다. 후보 생성과 로컬 분석은 그대로 유지되므로 사전 효과를 같은 조건에서
비교할 수 있습니다.

### 읽기 길이 정책

공백과 문장부호를 제외한 한글 음절을 기준으로 문장 길이를 평가합니다.

| 난이도 | 문장당 권장 음절 |
|---:|---:|
| 1 | 5~9 |
| 2 | 7~12 |
| 3 | 10~19 |
| 4 | 13~21 |
| 5 | 16~26 |

난이도 3의 짧은 이야기는 3~4문장, 전체 36~60음절을 권장합니다. 범위를 모두 지킨 후보는
20점을 받고, 긴 문장은 초과 음절당 8점, 전체 분량 초과는 음절당 4점을 추가로
감점합니다. 후보 선택 우선순위는 `회피 위반 최소 → 길이 통과 → 종합점수`입니다.
목표 특징도 짧은 이야기 전체 6회를 넘으면 초과 횟수마다 감점하여 목표 글자를
과도하게 반복한 글이 높은 점수를 받지 못하게 합니다.

## 비교 UI 실행

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-full-project-test\services\ai"

& ".\.venv\Scripts\uvicorn.exe" iread_ai.app:app `
  --host 127.0.0.1 --port 8083

& ".\.venv\Scripts\streamlit.exe" run .\training_set_review_app.py `
  --server.address 127.0.0.1 --server.port 8510
```

브라우저에서 `http://127.0.0.1:8510`을 열고 `X-API-Key`에
`AI_INTERNAL_API_KEY` 값을 입력합니다. `기존 방식과 함께 생성해 비교`를 켜면 같은
설정으로 `useLexicon=false/true`를 각각 실행합니다. 글 기반 훈련을 확인하려면
아동 수준을 `조금 어려워하는 아이` 이상으로 선택하고 `짧은 이야기 읽기` 또는
`짧은 글 읽기`를 반드시 포함할 훈련 유형에 추가합니다. 기초 자모 훈련은 규칙 DB가
직접 생성하므로 어휘 사전 적용 전후가 동일한 것이 정상입니다.

## 생성 파이프라인 연결

```text
Backend 학생 프로필
  → target / excluded / mastered 정책
  → AI 어휘 팔레트 조회
  → 규칙 기반 문항 조립 또는 LLM 프롬프트 어휘 후보
  → Kiwi·G2P 결과 검사
  → 통과 후보 반환
```

SQLite는 현재 조회량과 단일 AI 서버 배포에 충분합니다. 다중 인스턴스에서 동시 갱신이 필요해질 때 PostgreSQL로 옮기며, 의미 검색이 실제 요구될 때만 벡터 인덱스를 추가합니다. Elasticsearch는 이 단계에 필요하지 않습니다.
