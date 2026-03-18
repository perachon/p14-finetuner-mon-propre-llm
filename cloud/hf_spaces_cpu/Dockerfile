FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

# Copy app code (full context). This makes the Space more tolerant to layouts:
# - ./src/triage_llm (recommended)
# - ./triage_llm (fallback)
COPY . /app

# Hugging Face Spaces expects the app to listen on 7860
EXPOSE 7860

ENV TRIAGE_BACKEND=stub \
    PORT=7860

RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
