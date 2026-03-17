from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from triage_llm.schemas import DPORecord, MetadataSchema, SFTRecord
from triage_llm.utils import ensure_dir, read_jsonl, write_jsonl


def default_metadata_schema() -> MetadataSchema:
    return MetadataSchema(
        fields={
            "id": "Identifiant unique",
            "instruction/prompt": "Consigne ou prompt",
            "input": "Contexte (optionnel)",
            "output/chosen/rejected": "Réponse(s)",
            "symptoms": "Liste de symptômes normalisés (optionnel)",
            "history": "Antécédents (optionnel)",
            "vitals": "Constantes (optionnel)",
            "source": "Origine du dataset",
            "lang": "Langue fr/en",
            "confidence": "Niveau de confiance (optionnel)",
            "pii_redacted": "PII supprimées (bool)",
        }
    )


def load_records_from_dir(input_dir: Path) -> tuple[list[SFTRecord], list[DPORecord]]:
    sft: list[SFTRecord] = []
    dpo: list[DPORecord] = []

    for p in sorted(input_dir.glob("*.jsonl")):
        rows = read_jsonl(p)
        for row in rows:
            if {"instruction", "output"}.issubset(row.keys()):
                sft.append(SFTRecord.model_validate(row))
            elif {"prompt", "chosen", "rejected"}.issubset(row.keys()):
                dpo.append(DPORecord.model_validate(row))

    return sft, dpo


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
):
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = [rows[i] for i in idx[:n_train]]
    val = [rows[i] for i in idx[n_train : n_train + n_val]]
    test = [rows[i] for i in idx[n_train + n_val :]]
    return train, val, test


def build_datasets(input_dir: str, out_dir: str, seed: int = 42) -> dict[str, Path]:
    input_path = Path(input_dir)
    out_path = ensure_dir(out_dir)

    sft_records, dpo_records = load_records_from_dir(input_path)

    sft_rows = [r.model_dump(mode="json") for r in sft_records]
    dpo_rows = [r.model_dump(mode="json") for r in dpo_records]

    write_jsonl(out_path / "sft.jsonl", sft_rows)
    write_jsonl(out_path / "dpo.jsonl", dpo_rows)

    schema = default_metadata_schema()
    with open(out_path / "metadata_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    splits_path = ensure_dir(out_path / "splits")
    sft_train, sft_val, sft_test = split_rows(sft_rows, seed=seed)
    dpo_train, dpo_val, dpo_test = split_rows(dpo_rows, seed=seed)
    write_jsonl(splits_path / "sft_train.jsonl", sft_train)
    write_jsonl(splits_path / "sft_val.jsonl", sft_val)
    write_jsonl(splits_path / "sft_test.jsonl", sft_test)
    write_jsonl(splits_path / "dpo_train.jsonl", dpo_train)
    write_jsonl(splits_path / "dpo_val.jsonl", dpo_val)
    write_jsonl(splits_path / "dpo_test.jsonl", dpo_test)

    return {
        "sft": out_path / "sft.jsonl",
        "dpo": out_path / "dpo.jsonl",
        "schema": out_path / "metadata_schema.json",
        "splits": splits_path,
    }
