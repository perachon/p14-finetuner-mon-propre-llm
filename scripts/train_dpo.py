from __future__ import annotations

import argparse

from triage_llm.training.dpo import DPOConfig, run_dpo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dpo_jsonl", required=True)
    p.add_argument("--dpo_eval_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()

    cfg = DPOConfig(
        model_name_or_path=args.model_name_or_path,
        train_jsonl=args.dpo_jsonl,
        eval_jsonl=args.dpo_eval_jsonl,
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
    )
    run_dpo(cfg)


if __name__ == "__main__":
    main()
