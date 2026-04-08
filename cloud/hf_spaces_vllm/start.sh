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

# Optional: serve a LoRA adapter through vLLM.
#
# Expected env vars:
# - HF_TOKEN (if private)
# - VLLM_ENABLE_LORA=1
# - VLLM_LORA_REPO=perachon/p14-model
# - VLLM_LORA_SUBDIR=adapters/dpo_long_20260318_1657   (optional)
# - VLLM_LORA_NAME=triage                              (optional)
export VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-0}"
export VLLM_LORA_REPO="${VLLM_LORA_REPO:-}"
export VLLM_LORA_SUBDIR="${VLLM_LORA_SUBDIR:-}"
export VLLM_LORA_NAME="${VLLM_LORA_NAME:-triage}"

# Wire FastAPI backend to vLLM
export TRIAGE_BACKEND="vllm-openai"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}}"

echo "Starting vLLM on ${VLLM_HOST}:${VLLM_PORT} (model=${VLLM_MODEL})"

LORA_ARGS=()
if [[ "${VLLM_ENABLE_LORA}" == "1" && -n "${VLLM_LORA_REPO}" ]]; then
  echo "LoRA enabled: downloading ${VLLM_LORA_REPO}${VLLM_LORA_SUBDIR:+/${VLLM_LORA_SUBDIR}}"
  mkdir -p /app/lora
  python - <<'PY'
from __future__ import annotations

import os
import pathlib
import shutil

from huggingface_hub import snapshot_download

repo_id = os.environ["VLLM_LORA_REPO"]
subdir = os.environ.get("VLLM_LORA_SUBDIR", "").strip().strip("/")
local_dir = "/app/lora"

allow_patterns = None
if subdir:
    allow_patterns = [f"{subdir}/*"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
)

if subdir:
    src = pathlib.Path(local_dir) / subdir
    if not src.exists():
        raise RuntimeError(f"VLLM_LORA_SUBDIR not found in snapshot: {src}")
    for child in src.iterdir():
        shutil.move(str(child), local_dir)
PY

  LORA_ARGS=(
    --enable-lora
    --lora-modules "${VLLM_LORA_NAME}=/app/lora"
  )
fi

python -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${VLLM_MODEL}" \
  --dtype "auto" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests \
  "${LORA_ARGS[@]}" \
  &

# Wait a little for vLLM to start
sleep 3

echo "Starting FastAPI on ${API_HOST}:${API_PORT} (backend=${TRIAGE_BACKEND})"
exec python -m uvicorn triage_llm.api.app:app --host "${API_HOST}" --port "${API_PORT}" --app-dir src
