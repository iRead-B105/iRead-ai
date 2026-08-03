FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        gstreamer1.0-libav \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.32

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY assets ./assets

RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 iread

ENV PATH="/app/.venv/bin:${PATH}"

USER iread

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]

CMD ["uvicorn", "iread_ai.app:app", "--host", "0.0.0.0", "--port", "8080"]
