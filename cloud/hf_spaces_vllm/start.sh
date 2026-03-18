#!/usr/bin/env bash
set -euo pipefail

# Ports
# - Hugging Face Spaces expects the web app on 7860
# - vLLM OpenAI-compatible server runs on 8000 (internal)
export API_HOST="0.0.0.0"
export API_PORT="${API_PORT:-7860}"
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="${VLLM_PORT:-8000}"

# Model served by vLLM
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-1.7B-Base}"

# Wire FastAPI backend to vLLM
export TRIAGE_BACKEND="vllm-openai"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"

echo "Starting vLLM on ${VLLM_HOST}:${VLLM_PORT} (model=${VLLM_MODEL})"
python -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${VLLM_MODEL}" \
  --dtype "auto" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests \
  &

# Wait a little for vLLM to start
sleep 3

echo "Starting FastAPI on ${API_HOST}:${API_PORT} (backend=${TRIAGE_BACKEND})"
exec python -m uvicorn triage_llm.api.app:app --host "${API_HOST}" --port "${API_PORT}" --app-dir src
