from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triage_llm.data.build_datasets import BuildDatasetsConfig, build_datasets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anonymize", action="store_true", help="Active l'anonymisation Presidio")
    p.add_argument("--anonymize_lang_default", default="fr")
    p.add_argument(
        "--anonymize_operator",
        default="replace",
        choices=["replace", "mask", "redact"],
    )
    p.add_argument("--anonymize_new_value", default="<REDACTED>")
    p.add_argument(
        "--no_export_hf",
        action="store_true",
        help="Désactive l'export HF (save_to_disk)",
    )
    p.add_argument(
        "--clinical_eval_dir",
        default=None,
        help=(
            "Dossier séparé contenant des JSONL d'évaluation clinique "
            "(copié tel quel, sans mélange)"
        ),
    )
    args = p.parse_args()

    cfg = BuildDatasetsConfig(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        anonymize=args.anonymize,
        anonymize_lang_default=args.anonymize_lang_default,
        anonymize_operator=args.anonymize_operator,
        anonymize_new_value=args.anonymize_new_value,
        export_hf=not args.no_export_hf,
        clinical_eval_dir=args.clinical_eval_dir,
    )
    out = build_datasets(cfg)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
