# 교수자용 학습 프로필 분석 API

## 목적

`POST /api/v1/reports/analyze`는 Backend가 집계한 읽기 특성 프로필과 시선 추세를
교수자가 검토할 수 있는 관찰 문장으로 변환한다. 이 기능은 난독증 또는 학습장애를
진단하지 않으며, 원인·중증도·치료 방향을 생성하지 않는다.

## 처리 구조

1. `TeacherReportAnalyzer`가 입력 수치에서 근거 사실을 결정적으로 계산한다.
2. GMS가 설정된 경우 모델은 근거 ID를 인용하면서 사실을 짧은 문장으로 정리한다.
3. 모델 문장이 근거 범주, 수치, 변화 방향 또는 금지 표현 검증에 실패하면 전체 모델
   출력을 버리고 결정론적 문장으로 복구한다.
4. 응답의 `summaryProvider`로 `deterministic`, `gms`,
   `deterministic-fallback`을 구분한다.

## 요청 계약

요청은 `requestId`, `schemaVersion=1`, `profileAnalysisVersion`,
`featureProfiles`, `gazeTrend`로 구성된다.

`featureProfiles`는 Backend의 `student_feature_profiles` 집계 필드와 대응한다.

- 정확도, 발음 점수와 오류율
- 평균 고정 시간·횟수, 평균 회귀 횟수, 건너뜀 비율
- 평균 읽기 시간, 약점 점수, 신뢰도, 근거 수
- 선택적인 이전 정확도와 이전 약점 점수

공통 프로필 계약의 단위는 다음과 같다.

- `accuracyRate`, `pronunciationErrorRate`, `skipRate`, `weaknessScore`,
  `confidence`: `0.0~1.0`
- `avgPronunciationScore`: `0.0~100.0`
- `evidenceCount`, 시간·횟수 지표: 0 이상의 수
- `featureCode`: Backend `reading_features.feature_code` 형식
  (`GRAPHEME.*`, `SYLLABLE.*`, `PHONOLOGY.*`, `WORD.*`, `SENTENCE.*`)

`gazeTrend`는 보고서 스냅샷의 훈련·검사 시선 시계열과 대응한다. 원시 좌표, 원시
시선 파일, 음성, 이름, 학생 ID는 계약에 포함하지 않는다.

Backend `StudentFeatureProfileView`를 사용할 때는
`reading_profile_request_adapter.build_teacher_report_request`가 Backend 전용 필드인
`status`, `analysisVersion`, `analyzedAt`을 제거하고 보고서 요청을 만든다. 현재 Backend
뷰에는 교수자 표시용 특징 이름이 없으므로 `featureLabels` 매핑을 함께 제공해야 한다.
이전 정확도와 이전 약점 점수는 Backend가 별도 비교 스냅샷을 제공하기 전까지 생략한다.

## 응답 계약

응답은 기존 보고서 스냅샷에 매핑할 수 있는 다음 값을 반환한다.

- `improvedPatterns`
- `persistentDifficultyPatterns`
- `gazeDescriptions.training`
- `gazeDescriptions.test`

추가로 분석 규칙 버전, 요약 공급자, 데이터 충분도를 반환한다.

## V1 판정 기준

- 특성 근거 사용 최소 조건: 근거 3건 이상, 신뢰도 0.30 이상
- 향상: 이전 정확도 대비 10%p 이상 상승 또는 약점 점수 0.10 이상 감소
- 지속 관찰 필요: 정확도 60% 이하 또는 약점 점수 0.60 이상
- 노력형 성공: 정확도 80% 이상이면서 다음 중 하나 이상
  - 평균 고정 시간 1200ms 이상
  - 평균 고정 횟수 3회 이상
  - 평균 회귀 2회 이상
  - 평균 읽기 시간 2500ms 이상
- 시선 변화: 첫 값 대비 15% 이상 변한 체류 시간과 평균 체류 시간, 또는 역행 읽기
  횟수 변화를 관찰 사실로 기록

지속 관찰 항목이 5개를 넘으면 전체 약점 점수만으로 자르지 않는다. 가장 낮은 미숙달
단계의 항목을 최대 2개 먼저 포함하고, 남은 자리를 전체 우선순위가 높은 관찰 사실로
채운다. 따라서 교수자는 이후 단계의 큰 어려움과 현재 기초 단계에서 먼저 다룰 어려움을
함께 확인할 수 있다.

시선 지표의 증가·감소만으로 어려움의 원인을 단정하지 않는다. 수집 실패, 데이터
없음, 1회 관찰은 해석 보류 문장으로 반환한다.

교수자 문장은 향상 시 이전값·현재값·변화폭을 함께 보여 준다. 지속 어려움은 정확도,
종합 어려움 지표, 근거 수와 누적 프로필에 들어온 발음·시선 부담·읽기 시간 지표를
함께 보여 주고 다음 회기의 지속 관찰 필요성을 제시한다. 비교 가능한 시선 변화는
동일한 난이도에서 추가 확인할 필요성을 함께 제시한다. 모델이 이러한 핵심 의미나
근거 수치를 생략하면 결정론적 문장으로 복구한다.

근거 ID는 모델의 구조화 응답 `evidenceIds` 배열에만 기록하며 교수자에게 표시되는
`text`에는 노출하지 않는다. 모델이 본문에 내부 근거 ID를 포함하면 출력을 거부하고
결정론적 문장으로 복구한다.

## 실행 설정

- `AI_GENERATION_PROVIDER=mock`: 결정론적 분석과 문장만 사용
- `AI_GENERATION_PROVIDER=gms`와 `GMS_KEY`: 근거 기반 GMS 문장화를 시도

GMS 호출은 기존 `gpt-5.4-mini` Responses API 어댑터, `store=false`, strict JSON
Schema를 재사용한다. 공급자 오류와 검증 실패는 API 실패로 전파하지 않고
결정론적 문장으로 복구한다.

## 로컬 샘플 테스트 화면

Backend를 실행하지 않아도 AI 서버만으로 교수자 분석 결과를 확인할 수 있다.

```powershell
uv run uvicorn iread_ai.app:app --host 127.0.0.1 --port 8081
uv run streamlit run teacher_report_review_app.py --server.port 8508
```

브라우저에서 `http://127.0.0.1:8508`을 열고 샘플을 선택한 뒤 `AI 분석 실행`을
누른다. Docker UI만 실행하려면 다음 명령을 사용할 수 있다. AI API는 호스트의
8081 포트에서 별도로 실행되어 있어야 한다.

```powershell
docker compose -f compose.teacher-report-ui.yaml up --build
```

화면에는 다음 네 가지 익명 샘플이 포함된다.

- 꾸준히 성장하는 균형형: 향상·지속 어려움·시선 변화 동시 확인
- 정확하지만 많은 노력이 드는 유형: 높은 정답률 뒤의 읽기 부담 확인
- 근거 부족 신규 유형: 데이터 부족 시 판단 보류 확인
- 시선 보정 실패 유형: 실패 세션을 변화로 해석하지 않는지 확인

기본 수치는 Backend의
`src/main/resources/db/demo-data/teacher-personas.sql`에 있는 페르소나와
`student_feature_profiles`, `gaze_analysis_results` 시드 산식을 참고했다. 이전 기간
비교값과 판정 경계 테스트에 필요한 값은 샘플에서 명시적으로 보완했다. 샘플에는
이름, 학생 ID, 원시 음성, 원시 시선 좌표가 포함되지 않는다.

## Backend 연동 시 남은 일

현재 구현 범위는 AI 서버의 분석 API와 독립 실행용 검토 화면까지다. Backend와
Frontend는 수정하지 않았다. 추후 연동 담당자가 보고서 생성 과정에서 익명화된
집계 DTO로 이 API를 호출하고, 응답을 다음 필드에 연결해야 한다.

- `improvedPatterns` → `snapshot.improvedPatterns`
- `persistentDifficultyPatterns` → `snapshot.persistentDifficultyPatterns`
- `gazeDescriptions.training` → `snapshot.gazeTrend.training.descriptions`
- `gazeDescriptions.test` → `snapshot.gazeTrend.test.descriptions`
- 분석 버전·공급자·데이터 충분도와 현재 특징 기준값 → `snapshot.reportAnalysis`

기간 비교를 사용하려면 가장 최근의 이전 보고서 기준값을 `previousAccuracyRate`와
`previousWeaknessScore`로 전달해야 한다. 호출 실패 정책과 보고서 저장 방식은
Backend 연동 시 별도로 결정한다.
