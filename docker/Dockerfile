FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt pyproject.toml README.md /app/
COPY src/ /app/src/

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "triage_llm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
