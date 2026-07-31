FROM python:3.12-slim

# ffmpeg는 App이 올리는 WebM/Opus·MP4 녹음을 Azure Speech가 읽는 WAV PCM으로 변환한다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "iread_ai.app:app", "--host", "0.0.0.0", "--port", "8080"]
