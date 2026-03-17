from __future__ import annotations

import os
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
        return ModelBackendInfo(name="stub", details={"model_name_or_path": self.model_name_or_path})
