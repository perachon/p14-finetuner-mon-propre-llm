from __future__ import annotations

from pathlib import Path

from triage_llm.schemas import DPORecord, SFTRecord
from triage_llm.utils import read_jsonl


def test_sample_sft_validates() -> None:
    rows = read_jsonl(Path("data/sample/sft_sample.jsonl"))
    assert len(rows) >= 1
    for row in rows:
        SFTRecord.model_validate(row)


def test_sample_dpo_validates() -> None:
    rows = read_jsonl(Path("data/sample/dpo_sample.jsonl"))
    assert len(rows) >= 1
    for row in rows:
        DPORecord.model_validate(row)
