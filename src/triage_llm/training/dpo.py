from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer


@dataclass
class DPOConfig:
    model_name_or_path: str
    train_jsonl: str
    eval_jsonl: str | None
    output_dir: str
    beta: float = 0.1
    max_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42


def _map_dpo(example: dict) -> dict:
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def run_dpo(cfg: DPOConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)

    train_ds = Dataset.from_json(cfg.train_jsonl).map(_map_dpo, remove_columns=None)
    eval_ds = (
        Dataset.from_json(cfg.eval_jsonl).map(_map_dpo, remove_columns=None)
        if cfg.eval_jsonl
        else None
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.save_steps if eval_ds is not None else None,
        save_total_limit=3,
        seed=cfg.seed,
        fp16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        beta=cfg.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        max_prompt_length=int(cfg.max_length * 0.5),
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
