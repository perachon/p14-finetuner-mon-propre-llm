from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


@dataclass
class DPOConfig:
    model_name_or_path: str
    train_jsonl: str
    eval_jsonl: str | None
    output_dir: str
    beta: float = 0.1
    max_length: int = 256
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    max_steps: int | None = None
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42


def _torch_dtype(use_cuda: bool) -> torch.dtype | None:
    if not use_cuda:
        return None
    return torch.float16


def _load_policy_model_and_tokenizer(
    model_name_or_path: str,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load either a base model or a PEFT adapter directory.

    If `model_name_or_path` points to an adapter folder (contains adapter_config.json),
    load the base model then attach the adapter as trainable.
    """

    use_cuda = torch.cuda.is_available()
    dtype = _torch_dtype(use_cuda)

    model_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda and dtype is not None:
        # transformers v5 prefers `dtype`
        model_kwargs["dtype"] = dtype
        model_kwargs["attn_implementation"] = "sdpa"

    p = Path(model_name_or_path)
    if p.exists() and (p / "adapter_config.json").exists():
        adapter_cfg = json.loads((p / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = adapter_cfg.get("base_model_name_or_path")
        if not base_name:
            raise ValueError("adapter_config.json missing base_model_name_or_path")

        tokenizer = AutoTokenizer.from_pretrained(str(base_name), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(str(base_name), **model_kwargs)
        base_model.config.use_cache = False
        if use_cuda:
            try:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                base_model.gradient_checkpointing_enable()
            except Exception:
                pass
            torch.cuda.empty_cache()
            base_model = base_model.to("cuda")

        model = PeftModel.from_pretrained(base_model, str(p), is_trainable=True)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if use_cuda:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        torch.cuda.empty_cache()
        model = model.to("cuda")
    return model, tokenizer


def _map_dpo(example: dict) -> dict:
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def run_dpo(cfg: DPOConfig) -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    model, tokenizer = _load_policy_model_and_tokenizer(cfg.model_name_or_path)

    train_ds = Dataset.from_json(cfg.train_jsonl).map(_map_dpo, remove_columns=None)
    eval_ds = (
        Dataset.from_json(cfg.eval_jsonl).map(_map_dpo, remove_columns=None)
        if cfg.eval_jsonl
        else None
    )

    # TRL 0.29+ expects a TRL DPOConfig here.
    from trl.trainer.dpo_config import DPOConfig as TRLDPOConfig

    args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        seed=cfg.seed,
        fp16=use_cuda,
        bf16=False,
        report_to="none",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.save_steps if eval_ds is not None else None,
        per_device_eval_batch_size=1,
        max_length=cfg.max_length,
        use_cache=False,
        gradient_checkpointing=use_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=cfg.beta,
        precompute_ref_log_probs=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
