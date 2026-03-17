from __future__ import annotations

import inspect
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
    use_cuda = torch.cuda.is_available()
    fp16 = cfg.fp16 if cfg.fp16 is not None else use_cuda
    bf16 = cfg.bf16 if cfg.bf16 is not None else False

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

    if use_cuda and torch_dtype is not None:
        # transformers v5 prefers `dtype` (torch_dtype is deprecated)
        from_pretrained_params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        if "dtype" in from_pretrained_params:
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)

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
        "report_to": [],
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

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg.output_dir)
