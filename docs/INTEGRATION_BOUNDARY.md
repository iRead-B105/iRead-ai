# Backend·Orchestration 통합 경계

## 책임 분리

### Backend가 소유할 데이터

- 아동 식별자와 읽기 프로필 원본
- 이야기 세션, revision, 최근 페이지, 해결·미해결 사건
- 생성된 페이지와 이미지 저장 위치
- 훈련 결과와 프로필 버전 이력
- 사용자 권한, 감사 기록, 장기 멱등성

### AI 서버가 받을 데이터

- 비식별 `studentId` 라우팅 값
- Backend가 컴파일한 `generationProfile` 스냅샷과 버전·해시
- 현재 이야기 템플릿·장 계획
- 필요한 최소 이야기 상태와 직전 확정 입력

### AI 서버가 반환할 데이터

- 2~4페이지의 문장과 페이지별 `visualScene`
- Kiwi·G2P 기반 품질 분석
- 모델·프롬프트·후보·교정 provenance와 단계별 시간
- Backend가 revision 확인 후 반영할 `statePatch`
- 별도 요청으로 생성한 페이지 이미지

AI 서버가 Backend에 아동 프로필을 역으로 조회하지 않습니다. Backend가
요청 시점에 사용한 프로필 스냅샷을 함께 보내야 재현성, 감사 가능성,
서비스 간 장애 격리가 좋아집니다.

## 현재 저장소 간 차이

2026-07-31 기준:

- Orchestration 공통 OpenAPI에는 기존 v1 이야기 계약만 있습니다.
- Backend는 아직 v3 요청에 아동 프로필 스냅샷을 보내지 않습니다.
- Backend의 `story_scenes.image_url`은 현재 생성 흐름에서 채워지지 않습니다.
- Orchestration `compose.yml`은 실제 AI 컨테이너가 아니라 WireMock
  `ai-mock`을 사용합니다.
- Backend의 AI 읽기 제한 시간은 30초입니다.
- 승인된 제품 명세는 자유 STT 입력과 AI가 생성한 선택지 3개를 함께 지원하고
  `currentProgress/nextProgress`를 사용합니다. v3 후보 계약도 최종 페이지에
  서로 다른 선택지 3개를 반환하며, `storyRevision/statePatch`를 공식 계약에
  연결하는 작업은 별도로 진행합니다.

따라서 이 저장소의 v3·페이지 이미지 API는 현재 **AI 서비스 내부 후보
계약**입니다. 이번 기능 PR과 AI submodule 포인터 갱신만으로 실서비스
연동이 완료되었다고 볼 수 없습니다.

## 권장 후속 PR

1. Orchestration 계약 PR
   - v3 장 생성, `visualScene`, 페이지 이미지 스키마 확정
   - 자유 STT 입력을 공식 분기 입력으로 확정
2. Backend 어댑터 PR
   - 아동 프로필 스냅샷 컴파일·전달
   - `statePatch` revision 원자 반영
   - 글 우선 응답, 페이지 이미지 별도 병렬 요청·저장
3. Orchestration 런타임 PR
   - WireMock과 실제 AI 서비스를 명시적으로 선택하도록 Compose 수정
   - 공용 secret과 timeout 설정
4. 각 서비스 PR 병합 후 AI submodule 포인터 전용 PR

## 운영 전 해결할 항목

- 현재 멱등 저장소는 프로세스 메모리 기반입니다. 재시작·다중 worker·다중
  replica에서도 보장하려면 Redis 또는 DB 기반 공유 저장소가 필요합니다.
- 기존 훈련·이야기·이미지 mock 경로 일부는 런타임 인증이 없습니다.
  내부망 밖으로 노출하기 전에 인증 정책을 통일해야 합니다.
- 페이지 이미지 응답은 Base64입니다. 객체 저장소 업로드, URL 생성,
  `story_scenes.image_url` 저장은 Backend 책임입니다.
- v3의 내부 시간 예산은 글 28초, 조건부 교정 6초, 시각 장면 8초와 로컬
  처리를 합치면 Backend의 현재 30초 제한을 넘을 수 있습니다. 공통 계약
  PR에서 timeout 또는 비동기 처리 방식을 확정해야 합니다.
- 기본 캐릭터 레퍼런스는 저장소에 포함하지 않습니다. 에셋이 없어도
  `storyContext`와 `visualScene`만으로 그림을 생성하며, 선택적 레퍼런스는
  출처·사용 권한·버전을 확인한 파일만 서버에 배치해야 합니다.

## 공식 포인터 갱신 절차

AI 기능 PR이 `iRead-ai/develop`에 병합된 뒤 실행합니다.

```powershell
git switch develop
git pull --ff-only origin develop
git submodule update --init --recursive
git switch -c chore/update-ai-pointer

git -C services/ai fetch origin develop
git -C services/ai switch --detach origin/develop

git add services/ai
git diff --cached --submodule=log
git diff --cached --check
git commit -m "chore(submodule): ai develop 참조 갱신"
git push -u origin chore/update-ai-pointer
```

`chore/update-ai-pointer -> develop` PR에는 `services/ai` gitlink 변경만
포함합니다. 병합 후 Harness 성공과 GitLab Monorepo Sync 결과를 모두
확인해야 합니다.

서비스 저장소 `develop` 병합만으로 GitLab main에는 반영되지 않습니다.
