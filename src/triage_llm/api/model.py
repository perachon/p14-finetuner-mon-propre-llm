from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBackendInfo:
    name: str
    details: dict[str, Any]


class SimpleBackend:
    """Backend minimal pour POC.

    - En prod: remplacez par vLLM (in-process) ou un serveur vLLM OpenAI-compatible.
    - Ici: backend "stub" pour permettre de tester l'API sans GPU.
    """

    def __init__(self) -> None:
        self.model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return (
            "Je suis un POC de triage. Pour mieux évaluer, je dois poser quelques questions et "
            "je ne remplace pas un avis médical.\n\n"
            f"Résumé du message: {prompt[:200]}"
        )

    def info(self) -> ModelBackendInfo:
        return ModelBackendInfo(
            name="stub", details={"model_name_or_path": self.model_name_or_path}
        )


class TransformersPeftBackend:
    """Backend Transformers + PEFT (LoRA).

    Conçu pour être activé via variable d'env, et chargé *lazy* au 1er appel
    pour éviter de casser la CI/Docker (où torch CUDA et/ou les poids ne sont
    pas disponibles).
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        adapter_name_or_path: str | None,
    ) -> None:
        self.base_model_name_or_path = base_model_name_or_path
        self.adapter_name_or_path = adapter_name_or_path

        self._tokenizer = None
        self._model = None
        self._device = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )

        if self.adapter_name_or_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_name_or_path)

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _build_prompt(self, user_message: str, lang: str) -> str:
        # Minimal prompt: keep behavior stable; we do not attempt strict JSON parsing here.
        system_fr = (
            "Tu es un assistant de triage médical (POC éducatif). "
            "Tu dois être prudent, poser des questions, et proposer des étapes suivantes. "
            "N'invente pas de diagnostic. En cas de signes graves, recommande les urgences."
        )
        system_en = (
            "You are a medical triage assistant (educational POC). "
            "Be cautious, ask questions, and propose next steps. "
            "Do not invent a diagnosis. For red flags, recommend emergency care."
        )
        system = system_fr if lang == "fr" else system_en

        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return f"SYSTEM: {system}\nUSER: {user_message}\nASSISTANT:"

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        self._lazy_init()

        import torch

        assert self._model is not None
        assert self._tokenizer is not None

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        encoded = self._tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            encoded = {k: v.to(device) for k, v in encoded.items()}

        input_len = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            out = self._model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
            )

        new_tokens = out[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def info(self) -> ModelBackendInfo:
        device = self._device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = None
        return ModelBackendInfo(
            name="transformers-peft",
            details={
                "base_model": self.base_model_name_or_path,
                "adapter": self.adapter_name_or_path,
                "device": device,
            },
        )


class VllmOpenAIBackend:
    """Backend that calls a vLLM OpenAI-compatible server.

    Intended for cloud deployment where vLLM runs on Linux/GPU and this API
    simply forwards generation requests.

    Required env vars:
    - VLLM_BASE_URL (default: http://127.0.0.1:8000)
    - VLLM_MODEL (default: BASE_MODEL_NAME_OR_PATH or Qwen/Qwen3-1.7B-Base)
    Optional:
    - VLLM_API_KEY (default: empty)
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""

    def _system_prompt(self) -> str:
        # Keep it bilingual because the API contract currently calls `generate()`
        # without passing `lang`.
        return (
            "FR: Tu es un assistant de triage médical (POC éducatif). "
            "Sois prudent, pose des questions, propose des étapes suivantes. "
            "N'invente pas de diagnostic. En cas de signes graves, recommande les urgences.\n\n"
            "EN: You are a medical triage assistant (educational POC). "
            "Be cautious, ask questions, propose next steps. "
            "Do not invent a diagnosis. For red flags, recommend emergency care."
        )

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed: HTTP {e.code} - {err[:300]}") from e
        except Exception as e:
            raise RuntimeError(f"vLLM request failed: {type(e).__name__}: {e}") from e

        try:
            obj = json.loads(body)
            return (obj["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected vLLM response: {body[:300]}") from e

    def info(self) -> ModelBackendInfo:
        return ModelBackendInfo(
            name="vllm-openai",
            details={
                "base_url": self.base_url,
                "model": self.model,
            },
        )


def _default_adapter_path() -> str | None:
    # Prefer the latest known long-run adapter if present in this repo.
    candidates = [
        "checkpoints/qwen3-1.7b-dpo_LONG_20260318_1657",
        "checkpoints/qwen3-1.7b-dpo_from_sft_lowvram",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def make_backend_from_env() -> SimpleBackend | TransformersPeftBackend | VllmOpenAIBackend:
    backend = os.getenv("TRIAGE_BACKEND", "stub").strip().lower()
    if backend in {"stub", "simple", "noop"}:
        return SimpleBackend()

    if backend in {"transformers", "peft", "transformers-peft"}:
        base_model = os.getenv("BASE_MODEL_NAME_OR_PATH", "Qwen/Qwen3-1.7B-Base")
        adapter = os.getenv("ADAPTER_NAME_OR_PATH") or _default_adapter_path()
        return TransformersPeftBackend(base_model_name_or_path=base_model, adapter_name_or_path=adapter)

    if backend in {"vllm", "vllm-openai", "openai"}:
        base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
        model = os.getenv("VLLM_MODEL") or os.getenv("BASE_MODEL_NAME_OR_PATH", "Qwen/Qwen3-1.7B-Base")
        api_key = os.getenv("VLLM_API_KEY")
        return VllmOpenAIBackend(base_url=base_url, model=model, api_key=api_key)

    # Safe fallback
    return SimpleBackend()
