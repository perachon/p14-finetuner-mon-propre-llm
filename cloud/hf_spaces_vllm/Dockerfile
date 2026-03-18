# Hugging Face Spaces (Docker) + GPU
# Runs:
# - vLLM OpenAI-compatible server (internal port 8000)
# - FastAPI triage endpoint (public port 7860)

FROM vllm/vllm-openai:v0.6.2

WORKDIR /app

# Minimal deps for the FastAPI app (keep image smaller than installing full training stack)
COPY cloud/hf_spaces_vllm/requirements-api.txt /app/requirements-api.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-api.txt

# App source
COPY src/ /app/src/

ENV PYTHONPATH=/app/src
EXPOSE 7860

COPY cloud/hf_spaces_vllm/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
