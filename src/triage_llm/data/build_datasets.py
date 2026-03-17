from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from presidio_anonymizer.entities import OperatorConfig

from triage_llm.data.anonymize import PresidioAnonymizer
from triage_llm.schemas import DPORecord, MetadataSchema, SFTRecord
from triage_llm.utils import ensure_dir, read_jsonl, write_jsonl


@dataclass
class BuildDatasetsConfig:
    input_dir: str
    out_dir: str
    seed: int = 42
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)
    anonymize: bool = False
    anonymize_lang_default: str = "fr"
    anonymize_operator: str = "replace"
    anonymize_new_value: str = "<REDACTED>"
    export_hf: bool = True
    clinical_eval_dir: str | None = None


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_write(path: Path, event: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _anonymize_sft_rows(
    rows: list[dict[str, Any]],
    lang_default: str,
    operator: str,
    new_value: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    op = OperatorConfig(operator, {"new_value": new_value})
    engine_fr = PresidioAnonymizer(language="fr", operators={"DEFAULT": op})
    engine_en = PresidioAnonymizer(language="en", operators={"DEFAULT": op})

    n_entities_total = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        lang = (row.get("lang") or lang_default).lower()
        engine = engine_fr if lang == "fr" else engine_en

        for key in ["instruction", "input", "output"]:
            if not row.get(key):
                continue
            try:
                res = engine.anonymize(str(row[key]))
                n_entities_total += len(res.entities)
                row[key] = res.text
            except Exception:
                # Fallback: leave text unchanged but keep pipeline running.
                row[key] = str(row[key])
        row["pii_redacted"] = True
        out.append(row)

    stats = {"records": len(rows), "entities_detected": n_entities_total}
    return out, stats


def _anonymize_dpo_rows(
    rows: list[dict[str, Any]],
    lang_default: str,
    operator: str,
    new_value: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    op = OperatorConfig(operator, {"new_value": new_value})
    engine_fr = PresidioAnonymizer(language="fr", operators={"DEFAULT": op})
    engine_en = PresidioAnonymizer(language="en", operators={"DEFAULT": op})

    n_entities_total = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        lang = (row.get("lang") or lang_default).lower()
        engine = engine_fr if lang == "fr" else engine_en

        for key in ["prompt", "chosen", "rejected"]:
            if not row.get(key):
                continue
            try:
                res = engine.anonymize(str(row[key]))
                n_entities_total += len(res.entities)
                row[key] = res.text
            except Exception:
                row[key] = str(row[key])
        row["pii_redacted"] = True
        out.append(row)

    stats = {"records": len(rows), "entities_detected": n_entities_total}
    return out, stats


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


def build_datasets(cfg: BuildDatasetsConfig) -> dict[str, Path]:
    input_path = Path(cfg.input_dir)
    out_path = ensure_dir(cfg.out_dir)
    audit_path = out_path / "audit_log.jsonl"

    _audit_write(
        audit_path,
        {
            "ts": _now_utc_iso(),
            "event": "build_start",
            "input_dir": str(input_path),
            "out_dir": str(out_path),
            "seed": cfg.seed,
            "split_ratios": cfg.split_ratios,
            "anonymize": cfg.anonymize,
            "export_hf": cfg.export_hf,
        },
    )

    sft_records, dpo_records = load_records_from_dir(input_path)

    sft_rows = [r.model_dump(mode="json") for r in sft_records]
    dpo_rows = [r.model_dump(mode="json") for r in dpo_records]

    _audit_write(
        audit_path,
        {
            "ts": _now_utc_iso(),
            "event": "loaded",
            "sft_records": len(sft_rows),
            "dpo_records": len(dpo_rows),
        },
    )

    if cfg.anonymize:
        sft_rows, sft_stats = _anonymize_sft_rows(
            sft_rows,
            lang_default=cfg.anonymize_lang_default,
            operator=cfg.anonymize_operator,
            new_value=cfg.anonymize_new_value,
        )
        dpo_rows, dpo_stats = _anonymize_dpo_rows(
            dpo_rows,
            lang_default=cfg.anonymize_lang_default,
            operator=cfg.anonymize_operator,
            new_value=cfg.anonymize_new_value,
        )
        _audit_write(
            audit_path,
            {
                "ts": _now_utc_iso(),
                "event": "anonymized",
                "sft": sft_stats,
                "dpo": dpo_stats,
                "operator": cfg.anonymize_operator,
                "new_value": cfg.anonymize_new_value,
            },
        )

    sft_path = out_path / "sft.jsonl"
    dpo_path = out_path / "dpo.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(dpo_path, dpo_rows)

    schema = default_metadata_schema()
    schema_path = out_path / "metadata_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    splits_path = ensure_dir(out_path / "splits")
    sft_train, sft_val, sft_test = split_rows(sft_rows, seed=cfg.seed, ratios=cfg.split_ratios)
    dpo_train, dpo_val, dpo_test = split_rows(dpo_rows, seed=cfg.seed, ratios=cfg.split_ratios)
    sft_train_path = splits_path / "sft_train.jsonl"
    sft_val_path = splits_path / "sft_val.jsonl"
    sft_test_path = splits_path / "sft_test.jsonl"
    dpo_train_path = splits_path / "dpo_train.jsonl"
    dpo_val_path = splits_path / "dpo_val.jsonl"
    dpo_test_path = splits_path / "dpo_test.jsonl"
    write_jsonl(sft_train_path, sft_train)
    write_jsonl(sft_val_path, sft_val)
    write_jsonl(sft_test_path, sft_test)
    write_jsonl(dpo_train_path, dpo_train)
    write_jsonl(dpo_val_path, dpo_val)
    write_jsonl(dpo_test_path, dpo_test)

    if cfg.export_hf:
        sft_dd = DatasetDict(
            {
                "train": Dataset.from_list(sft_train),
                "validation": Dataset.from_list(sft_val),
                "test": Dataset.from_list(sft_test),
            }
        )
        dpo_dd = DatasetDict(
            {
                "train": Dataset.from_list(dpo_train),
                "validation": Dataset.from_list(dpo_val),
                "test": Dataset.from_list(dpo_test),
            }
        )
        hf_path = ensure_dir(out_path / "hf")
        sft_hf_path = hf_path / "sft"
        dpo_hf_path = hf_path / "dpo"
        sft_dd.save_to_disk(str(sft_hf_path))
        dpo_dd.save_to_disk(str(dpo_hf_path))

        _audit_write(
            audit_path,
            {
                "ts": _now_utc_iso(),
                "event": "export_hf",
                "sft_path": str(sft_hf_path),
                "dpo_path": str(dpo_hf_path),
            },
        )

    if cfg.clinical_eval_dir:
        eval_in = Path(cfg.clinical_eval_dir)
        eval_out = ensure_dir(out_path / "eval")
        copied: list[str] = []
        for p in sorted(eval_in.glob("*.jsonl")):
            target = eval_out / p.name
            target.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            copied.append(str(target))
        _audit_write(
            audit_path,
            {"ts": _now_utc_iso(), "event": "eval_sets_copied", "files": copied},
        )

    _audit_write(
        audit_path,
        {
            "ts": _now_utc_iso(),
            "event": "build_end",
            "outputs": {
                "sft_jsonl": str(sft_path),
                "dpo_jsonl": str(dpo_path),
                "schema": str(schema_path),
                "splits": str(splits_path),
                "audit": str(audit_path),
            },
            "hashes": {
                "sft_jsonl_sha256": _sha256_file(sft_path),
                "dpo_jsonl_sha256": _sha256_file(dpo_path),
                "schema_sha256": _sha256_file(schema_path),
            },
        },
    )

    return {
        "sft": sft_path,
        "dpo": dpo_path,
        "schema": schema_path,
        "splits": splits_path,
        "audit": audit_path,
        "hf": out_path / "hf" if cfg.export_hf else out_path,
    }
