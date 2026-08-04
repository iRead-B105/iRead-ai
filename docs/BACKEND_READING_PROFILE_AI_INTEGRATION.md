# Backend 읽기 프로필 기반 AI API 연동안

## 범위

이 문서는 Backend가 이미 계산해 저장하는 `student_feature_profiles`를 다음 두 AI API에
전달하기 위한 계약 초안이다.

- `POST /api/v1/reports/analyze`: 교수자용 학습 관찰 요약
- `POST /api/v1/curricula/recommend`: 다음 학습일 훈련 5개 추천

이 연동은 이야기·교안·훈련 문항 생성 계약을 변경하지 않는다. AI 서버는 Backend DB를
직접 조회하지 않고, Backend가 요청 시점에 만든 집계 스냅샷만 받는다.

## 기준 데이터

Backend `StudentFeatureProfileService.StudentFeatureProfileView`가 프로필 수치의 기준이다.

| AI 공통 필드 | Backend 원천 | 단위 | 필수 여부 |
| --- | --- | --- | --- |
| `featureCode` | `reading_features.feature_code` | Backend 특징 코드 | 필수 |
| `accuracyRate` | `accuracyRate` | `0.0~1.0` | 필수 |
| `avgPronunciationScore` | `avgPronunciationScore` | `0.0~100.0` | 선택 |
| `avgFixationDurationMs` | `avgFixationDurationMs` | ms | 선택 |
| `avgFixationCount` | `avgFixationCount` | 횟수 | 선택 |
| `avgRegressionCount` | `avgRegressionCount` | 횟수 | 선택 |
| `skipRate` | `skipRate` | `0.0~1.0` | 필수 |
| `avgReadingTimeMs` | `avgReadingTimeMs` | ms | 선택 |
| `weaknessScore` | `weaknessScore` | `0.0~1.0` | 필수 |
| `confidence` | `confidence` | `0.0~1.0` | 필수 |
| `evidenceCount` | `evidenceCount` | 건수 | 필수 |

Backend 조회용 `status`, `analysisVersion`, `analyzedAt`은 AI 기능 공통 수치가 아니므로
각 기능 요청으로 변환할 때 필요한 위치에만 사용한다.

특징 코드는 다음 네임스페이스만 사용한다.

- `GRAPHEME.*`
- `SYLLABLE.*`
- `PHONOLOGY.*`
- `WORD.*`
- `SENTENCE.*`

## 교수자 분석 요청

Backend는 프로필과 시선 추세를 결합해 다음 형태로 요청한다.

```json
{
  "requestId": "teacher-report-<uuid>",
  "schemaVersion": 1,
  "profileAnalysisVersion": "WEAKNESS_V1",
  "featureProfiles": [
    {
      "featureCode": "SYLLABLE.COMPLEX_CODA",
      "featureLabel": "겹받침 음절 읽기",
      "accuracyRate": 0.42,
      "avgPronunciationScore": 54.0,
      "avgFixationDurationMs": 1350,
      "avgFixationCount": 3.1,
      "avgRegressionCount": 2.4,
      "skipRate": 0.18,
      "avgReadingTimeMs": 2800,
      "weaknessScore": 0.73,
      "confidence": 0.82,
      "evidenceCount": 15
    }
  ],
  "gazeTrend": {
    "training": {
      "status": "NO_DATA",
      "comparisonAvailable": false,
      "points": [],
      "failedSessionCount": 0
    },
    "test": {
      "status": "NO_DATA",
      "comparisonAvailable": false,
      "points": [],
      "failedSessionCount": 0
    }
  }
}
```

### Backend에서 추가로 필요한 값

현재 `StudentFeatureProfileView`에는 `featureLabel`이 없다. DB와
`ReadingFeatureEntity.featureName`에는 이미 값이 있으므로 새 테이블 없이 View에
`featureName`을 포함하거나 요청 조립 시 읽기 특징 엔티티에서 가져오면 된다.

`previousAccuracyRate`, `previousWeaknessScore`는 선택 필드다. 과거 스냅샷이 없으면
생략하며, 이 경우 AI는 향상 여부를 만들지 않고 현재의 지속 관찰 특징만 설명한다.

## 커리큘럼 추천 요청

Backend는 같은 프로필에 최근 훈련 이력을 결합한다.

```json
{
  "requestId": "curriculum-<uuid>",
  "schemaVersion": 1,
  "featureProfiles": [
    {
      "featureCode": "SYLLABLE.COMPLEX_CODA",
      "accuracyRate": 0.42,
      "avgPronunciationScore": 54.0,
      "avgFixationDurationMs": 1350,
      "avgFixationCount": 3.1,
      "avgRegressionCount": 2.4,
      "skipRate": 0.18,
      "avgReadingTimeMs": 2800,
      "weaknessScore": 0.73,
      "confidence": 0.82,
      "evidenceCount": 15
    }
  ],
  "recentTrainings": [
    {
      "trainingTemplateId": 22,
      "accuracy": 0.58,
      "daysAgo": 1
    }
  ],
  "useLlm": true
}
```

`category`는 선택 필드다. 보내는 경우 반드시 `featureCode`의 첫 네임스페이스와 같아야
한다. 생략하면 AI 서버가 `featureCode`에서 계산한다.

## 오류와 하위 호환성

- `X-API-Key`와 `Idempotency-Key`는 필수다.
- `Idempotency-Key`는 요청의 `requestId`와 같아야 한다.
- 비율에 `73`, `730` 등 비정규화 값을 보내면 `400`으로 거부한다.
- 발음 점수에 `540`처럼 `100`을 넘는 값을 보내면 `400`으로 거부한다.
- Backend 네임스페이스가 아닌 특징 코드는 `400`으로 거부한다.
- 교수자 분석의 LLM이 실패하면 근거 기반 결정론적 요약으로 복구한다.
- 커리큘럼 LLM이 실패하면 단계 제한이 적용된 결정론적 추천으로 복구한다.

## 최소 Backend 연결 작업

AI 서버 검증이 끝난 뒤 Backend에는 다음 연결만 필요하다.

1. `StudentFeatureProfileView`에서 프로필 요청 DTO 조립
2. 교수자 요청에 `reading_features.feature_name`과 시선 추세 결합
3. 커리큘럼 요청에 최근 훈련 템플릿 ID·정확도·경과 일수 결합
4. `AiClient`에 두 엔드포인트 호출 메서드 추가
5. 응답 저장 또는 기존 교수자·커리큘럼 서비스에 매핑

DB 스키마 변경, AI 서버의 Backend DB 직접 접근, 원시 음성·원시 시선 좌표 전송은
필요하지 않다.

## AI 서버 검증 상태

- Backend `StudentFeatureProfileView`와 같은 익명 스냅샷 계약 검증
- 같은 스냅샷에서 교수자 분석·커리큘럼 요청 생성 검증
- 두 HTTP 엔드포인트 통합 테스트
- 전체 테스트와 Ruff 검사

테스트용 스냅샷은 `iread_ai.devtools.backend_profile_samples`, 변환기는
`iread_ai.application.reading_profile_request_adapter`에 있다.
