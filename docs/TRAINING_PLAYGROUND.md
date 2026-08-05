# 훈련 통합 테스트 UI

`training_playground_app.py`는 제품 프론트엔드가 아니라 AI 커리큘럼과 교안의
품질을 확인하는 개발용 Streamlit 앱이다. 운영 DB에 등록된 31개 훈련 형식을
직접 풀어볼 수 있다.

## 제공 기능

1. 아동 프로필 프리셋 선택
2. `POST /api/v1/curricula/recommend`로 다음 회차 훈련 5개 추천
3. 추천된 각 훈련에 `POST /api/v1/trainings/candidates`를 호출해 문항 5개 생성
4. 훈련 유형에 맞는 선택·조합·읽기·녹음 화면으로 문항 체험
5. 정답, 진행률, provider, 생성 시간, 요청·응답 JSON 확인
6. 전체 훈련 목록에서 특정 훈련 하나만 골라 재생성

이미지 생성 API와 발음 평가 API는 자동 호출하지 않는다. 녹음 위젯은 화면
흐름과 필수 입력을 확인하기 위한 용도다.

## 실행

먼저 통합 프로젝트의 AI 서버가 `http://127.0.0.1:8081`에서 실행 중이어야 한다.

```powershell
cd "C:\Users\SSAFY\Documents\New project\iRead-e2e-latest\services\ai"
docker compose -f compose.training-playground-ui.yaml up -d --build
docker compose -f compose.training-playground-ui.yaml ps
```

브라우저에서 `http://127.0.0.1:8511`을 연다.

로그는 다음 명령으로 확인한다.

```powershell
docker compose -f compose.training-playground-ui.yaml logs -f ui
```

종료할 때는 다음 명령을 실행한다.

```powershell
docker compose -f compose.training-playground-ui.yaml down
```

## 테스트 순서

1. 사이드바에서 아동 프로필을 선택한다.
2. `커리큘럼 생성`을 누른다.
3. 추천 역할, 난이도, 목표 특징과 provider를 확인한다.
4. `추천 교안 5개 생성`을 누른다.
5. 교안 표에서 5문항, provider, 생성 시간과 오류 여부를 확인한다.
6. `훈련 체험` 탭에서 훈련과 문항을 바꾸며 직접 답한다.
7. 현재 문항 진단에서 생성 원문과 목표 특징을 확인한다.

LLM 커리큘럼과 문장형 교안 생성은 API 비용이 발생할 수 있다. 규칙·사전형
훈련은 `rule-db` provider가 표시된다.
