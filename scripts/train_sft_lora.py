from __future__ import annotations

import argparse

from triage_llm.training.sft_lora import SFTConfig, run_sft_lora


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--sft_jsonl", required=True)
    p.add_argument("--sft_eval_jsonl", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_seq_length", type=int, default=1024)
    args = p.parse_args()

    cfg = SFTConfig(
        model_name_or_path=args.model_name_or_path,
        train_jsonl=args.sft_jsonl,
        eval_jsonl=args.sft_eval_jsonl,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
    )
    run_sft_lora(cfg)


if __name__ == "__main__":
    main()
