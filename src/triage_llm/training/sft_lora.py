from __future__ import annotations

from dataclasses import dataclass

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


def _format_example(example: dict) -> dict:
    instruction = example["instruction"]
    inp = example.get("input")
    output = example["output"]
    if inp:
        text = (
            f"### Instruction\n{instruction}\n\n### Input\n{inp}\n\n### Response\n{output}"
        )
    else:
        text = f"### Instruction\n{instruction}\n\n### Response\n{output}"
    return {"text": text}


def run_sft_lora(cfg: SFTConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_ds = Dataset.from_json(cfg.train_jsonl).map(_format_example, remove_columns=None)
    eval_ds = (
        Dataset.from_json(cfg.eval_jsonl).map(_format_example, remove_columns=None)
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
        bf16=False,
        fp16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=args,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
