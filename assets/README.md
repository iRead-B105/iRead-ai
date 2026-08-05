# 선택적 캐릭터 레퍼런스

페이지 간 캐릭터 외형을 더 강하게 유지하고 싶을 때만 승인된 PNG·JPEG·WebP
파일을 `character-references/`에 둡니다. 파일이 없어도 이미지 API는
`storyContext`와 `visualScene`을 이용해 정상적으로 생성합니다.

기본 파일명은 다음과 같습니다.

| `characterId` | 파일명 |
|---|---|
| `hare` | `rabbit.png` |
| `tortoise` | `turtle.png` |

저장소에는 출처와 사용 권한이 확인된 파일만 추가하세요. 파일을 추가하는 PR에는
제작자 또는 생성 도구, 라이선스·사용 범위, 확인 날짜를 함께 기록해야 합니다.
