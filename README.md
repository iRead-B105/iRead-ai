# iRead AI

iRead의 AI 서비스 저장소입니다.

## 기술 스택

- FastAPI
- Python 3.12
- uv
- Azure Speech SDK

## 발음 평가

`POST /api/v1/speech/pronunciation/analyze`는 30초 미만 한국어 읽기
녹음과 기준 문장을 받아 Azure Speech scripted Pronunciation Assessment를
실행한다.

- locale: `ko-KR`
- grading system: `HundredMark`
- granularity: `Word`
- miscue: 활성화
- 음성은 임시 파일로만 사용하고 요청 종료 시 삭제
- Azure 자격증명과 원본 응답은 응답·로그에 노출하지 않음

WAV는 8/16kHz, 16-bit mono PCM을 기본 입력으로 사용한다. Backend가 허용하는
WebM·MP3·MP4/M4A 같은 압축 음성을 그대로 처리하려면 AI server 실행 환경에
Azure Speech SDK와 같은 아키텍처의 GStreamer 런타임과 플러그인이 설치되어
있어야 한다.

```powershell
Copy-Item .env.example .env
uv sync --extra dev
uv run uvicorn iread_ai.app:app --host 0.0.0.0 --port 8081
uv run pytest
```

환경변수는 `.env.example`을 참고한다. Backend의 `AI_API_KEY`와 AI server의
`AI_INTERNAL_API_KEY`는 같은 값을 사용한다.
