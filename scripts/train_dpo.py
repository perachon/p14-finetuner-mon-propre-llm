from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Optional: load environment variables from a local .env (kept out of git)
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ROOT / ".env")
except Exception:
    pass

from triage_llm.training.dpo import DPOConfig, run_dpo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dpo_jsonl", required=True)
    p.add_argument("--dpo_eval_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = DPOConfig(
        model_name_or_path=args.model_name_or_path,
        train_jsonl=args.dpo_jsonl,
        eval_jsonl=args.dpo_eval_jsonl,
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    run_dpo(cfg)


if __name__ == "__main__":
    main()
