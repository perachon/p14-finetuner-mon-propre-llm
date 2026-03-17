from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage_llm.training.sft_lora import SFTConfig, run_sft_lora


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=False)
    p.add_argument("--sft_jsonl", required=True)
    p.add_argument("--sft_eval_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="Liste des modules LoRA (optionnel). Exemple GPT2: c_attn c_proj",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_trust_remote_code", action="store_true")
    p.add_argument("--fp16", action="store_true", help="Force fp16 (GPU)")
    p.add_argument("--no_fp16", action="store_true", help="Désactive fp16")
    p.add_argument("--bf16", action="store_true", help="Force bf16 (GPU)")
    p.add_argument("--no_bf16", action="store_true", help="Désactive bf16")
    p.add_argument(
        "--sanity_tiny",
        action="store_true",
        help="Preset: tiny model CPU-friendly pour valider la pipeline (5 steps)",
    )
    args = p.parse_args()

    model_name_or_path = args.model_name_or_path
    trust_remote_code = True
    if args.no_trust_remote_code:
        trust_remote_code = False
    elif args.trust_remote_code:
        trust_remote_code = True

    fp16 = None
    if args.fp16:
        fp16 = True
    if args.no_fp16:
        fp16 = False

    bf16 = None
    if args.bf16:
        bf16 = True
    if args.no_bf16:
        bf16 = False

    max_steps = args.max_steps
    target_modules = args.target_modules
    if args.sanity_tiny:
        model_name_or_path = model_name_or_path or "sshleifer/tiny-gpt2"
        trust_remote_code = False
        max_steps = max_steps or 5
        target_modules = target_modules or ["c_attn", "c_proj"]
        fp16 = False
        bf16 = False

    if not model_name_or_path:
        raise SystemExit(
            "--model_name_or_path est requis (sauf si --sanity_tiny fournit un preset)"
        )

    cfg = SFTConfig(
        model_name_or_path=model_name_or_path,
        train_jsonl=args.sft_jsonl,
        eval_jsonl=args.sft_eval_jsonl,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        trust_remote_code=trust_remote_code,
        fp16=fp16,
        bf16=bf16,
    )
    run_sft_lora(cfg)


if __name__ == "__main__":
    main()
