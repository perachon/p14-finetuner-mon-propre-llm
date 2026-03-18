#!/usr/bin/env sh
set -eu

# HF Spaces usually sets PORT=7860, but we default it anyway.
PORT="${PORT:-7860}"

# Auto-detect app dir.
APP_DIR="/app"
if [ -d "/app/src/triage_llm" ]; then
  APP_DIR="/app/src"
elif [ -d "/app/triage_llm" ]; then
  APP_DIR="/app"
fi

# Use --app-dir so imports like triage_llm.* resolve.
exec uvicorn triage_llm.api.app:app \
  --app-dir "${APP_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --proxy-headers \
  --forwarded-allow-ips '*'
