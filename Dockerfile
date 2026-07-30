FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "iread_ai.app:app", "--host", "0.0.0.0", "--port", "8080"]
