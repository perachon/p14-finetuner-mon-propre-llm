from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_jsonl: str
    eval_jsonl: str | None
    output_dir: str
    max_seq_length: int = 1024
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    trust_remote_code: bool = True
    fp16: bool | None = None
    bf16: bool | None = None
    report_to: str = "mlflow"
    run_name: str | None = None


def _normalize_report_to(value: str | None) -> str | list[str]:
    if value is None:
        return "none"
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "off", "disabled"}:
        return "none"
    return [normalized]


def _configure_tracking_environment(report_to: str, output_dir: str) -> None:
    normalized = report_to.strip().lower()
    if normalized != "mlflow":
        return

    if not os.getenv("MLFLOW_TRACKING_URI"):
        tracking_dir = os.path.join(os.path.dirname(output_dir), "mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{os.path.abspath(tracking_dir)}"

    if not os.getenv("MLFLOW_EXPERIMENT_NAME"):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "p14-triage-llm"


def _format_text(example: dict) -> str:
    instruction = str(example.get("instruction", ""))
    inp = example.get("input")
    output = str(example.get("output", ""))
    if inp:
        return (
            f"### Instruction\n{instruction}\n\n### Input\n{inp}\n\n### Response\n{output}"
        )
    return f"### Instruction\n{instruction}\n\n### Response\n{output}"


def run_sft_lora(cfg: SFTConfig) -> None:
    _configure_tracking_environment(cfg.report_to, cfg.output_dir)

    use_cuda = torch.cuda.is_available()
    fp16 = cfg.fp16 if cfg.fp16 is not None else use_cuda
    bf16 = cfg.bf16 if cfg.bf16 is not None else False

    if use_cuda:
        # Prefer SDPA when available (often lower memory than eager attention).
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    torch_dtype = None
    if use_cuda:
        if bf16:
            torch_dtype = torch.bfloat16
        elif fp16:
            torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.max_seq_length

    model_kwargs: dict[str, object] = {
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if use_cuda:
        # Try to use PyTorch SDPA attention when supported by the model.
        model_kwargs["attn_implementation"] = "sdpa"

    if use_cuda and torch_dtype is not None:
        # transformers v5 prefers `dtype` (torch_dtype is deprecated)
        from_pretrained_params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        if "dtype" in from_pretrained_params:
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)

    # Reduce memory usage during training.
    model.config.use_cache = False
    if use_cuda:
        try:
            # Prefer non-reentrant checkpointing when available (less overhead).
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        torch.cuda.empty_cache()
        # Move model to GPU early (before Trainer init) to reduce peak allocations.
        model = model.to("cuda")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.target_modules,
    )

    train_ds_raw = Dataset.from_json(cfg.train_jsonl)
    eval_ds_raw = Dataset.from_json(cfg.eval_jsonl) if cfg.eval_jsonl else None

    # Backward compat: some TRL versions expect a dataset with a `text` field.
    # Newer TRL (>=0.29) expects `formatting_func`.
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    uses_dataset_text_field = "dataset_text_field" in trainer_params
    if uses_dataset_text_field:
        train_ds = train_ds_raw.map(lambda ex: {"text": _format_text(ex)})
        eval_ds = eval_ds_raw.map(lambda ex: {"text": _format_text(ex)}) if eval_ds_raw else None
    else:
        train_ds = train_ds_raw
        eval_ds = eval_ds_raw

    ta_kwargs: dict[str, object] = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.save_steps if eval_ds is not None else None,
        "save_total_limit": 3,
        "seed": cfg.seed,
        "bf16": bf16,
        "fp16": fp16,
        "report_to": _normalize_report_to(cfg.report_to),
        "run_name": cfg.run_name or os.path.basename(cfg.output_dir),
    }

    # transformers v4 uses `evaluation_strategy`, v5 may use `eval_strategy`
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_value = "steps" if eval_ds is not None else "no"
    if "evaluation_strategy" in params:
        ta_kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in params:
        ta_kwargs["eval_strategy"] = strategy_value

    args = TrainingArguments(**ta_kwargs)

    trainer_kwargs: dict[str, object] = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": lora_config,
    }

    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if uses_dataset_text_field:
        trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in trainer_params:
            trainer_kwargs["max_seq_length"] = cfg.max_seq_length
    else:
        # TRL 0.29+ API
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        if "formatting_func" in trainer_params:
            trainer_kwargs["formatting_func"] = _format_text

    # TRL 0.29+ uses its own SFTConfig class (adds max_length, packing, etc.).
    # If available, prefer it to ensure truncation/padding behavior is applied.
    trl_args = None
    try:
        from trl.trainer.sft_config import SFTConfig as TRLSFTConfig

        if "trl.trainer.sft_config.SFTConfig" in str(trainer_params.get("args")):
            trl_args = TRLSFTConfig(
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
                bf16=bf16,
                fp16=fp16,
                report_to=cfg.report_to,
                run_name=cfg.run_name or os.path.basename(cfg.output_dir),
                eval_strategy="steps" if eval_ds is not None else "no",
                eval_steps=cfg.save_steps if eval_ds is not None else None,
                per_device_eval_batch_size=1,
                use_cache=False,
                gradient_checkpointing=use_cuda,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                max_length=cfg.max_seq_length,
                packing=False,
            )
    except Exception:
        trl_args = None

    if trl_args is not None:
        trainer_kwargs["args"] = trl_args

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg.output_dir)
