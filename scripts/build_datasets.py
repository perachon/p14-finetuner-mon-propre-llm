from __future__ import annotations

import argparse

from triage_llm.data.build_datasets import build_datasets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = build_datasets(input_dir=args.input_dir, out_dir=args.out_dir, seed=args.seed)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
