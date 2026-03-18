from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # Keep it minimal; builder will validate against Pydantic schemas.
            f.write(__import__("json").dumps(r, ensure_ascii=False) + "\n")


def _iter_all_splits(dd: DatasetDict) -> Dataset:
    parts: list[Dataset] = []
    for split in dd.keys():
        parts.append(dd[split])
    if not parts:
        raise ValueError("No splits found")
    if len(parts) == 1:
        return parts[0]
    return __import__("datasets").concatenate_datasets(parts)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Ingest cyrille-elie/CHSA-Triage-Medic-Full-Dataset from HF and convert "
            "to this repo JSONL schema."
        )
    )
    p.add_argument(
        "--out_dir",
        default="data/raw/chsa_hf",
        help="Output directory where JSONL files will be written.",
    )
    p.add_argument(
        "--max_sft",
        type=int,
        default=None,
        help="Optional cap on number of SFT examples (for quick tests).",
    )
    p.add_argument(
        "--max_dpo",
        type=int,
        default=None,
        help="Optional cap on number of DPO examples (for quick tests).",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "cyrille-elie/CHSA-Triage-Medic-Full-Dataset"

    sft_dd = load_dataset(dataset_id, "sft_medical_dataset")
    dpo_dd = load_dataset(dataset_id, "dpo_dataset")

    sft_all = _iter_all_splits(sft_dd)
    dpo_all = _iter_all_splits(dpo_dd)

    sft_rows: list[dict] = []
    for i, ex in enumerate(sft_all):
        if args.max_sft is not None and len(sft_rows) >= args.max_sft:
            break
        lang = str(ex.get("language") or "").strip().lower()
        if lang not in {"fr", "en"}:
            # Default to FR if unknown; but dataset is tagged fr/en.
            lang = "fr"
        src = str(ex.get("source_dataset") or dataset_id).strip()
        sft_rows.append(
            {
                "id": f"chsa_sft_{i}",
                "instruction": str(ex.get("instruction") or "").strip(),
                "input": None,
                "output": str(ex.get("response") or "").strip(),
                "lang": lang,
                "source": src,
                "pii_redacted": True,
                "created_at": _now_utc_iso(),
            }
        )

    dpo_rows: list[dict] = []
    for i, ex in enumerate(dpo_all):
        if args.max_dpo is not None and len(dpo_rows) >= args.max_dpo:
            break
        prompt = str(ex.get("prompt") or "").strip()
        # This config does not ship a language column; samples are English.
        lang = "en"
        dpo_rows.append(
            {
                "id": f"chsa_dpo_{i}",
                "prompt": prompt,
                "chosen": str(ex.get("chosen") or "").strip(),
                "rejected": str(ex.get("rejected") or "").strip(),
                "lang": lang,
                "source": f"{dataset_id}/dpo_dataset",
                "pii_redacted": True,
                "created_at": _now_utc_iso(),
            }
        )

    sft_path = out_dir / "sft_chsa.jsonl"
    dpo_path = out_dir / "dpo_chsa.jsonl"
    _write_jsonl(sft_path, sft_rows)
    _write_jsonl(dpo_path, dpo_rows)

    print(f"Wrote SFT: {sft_path} (rows={len(sft_rows)})")
    print(f"Wrote DPO: {dpo_path} (rows={len(dpo_rows)})")


if __name__ == "__main__":
    main()
