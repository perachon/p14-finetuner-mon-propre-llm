from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    license: str | None
    languages: list[str]
    tags: list[str]
    likes: int | None
    downloads: int | None
    last_modified: str | None


def _normalize_lang_tag(tag: str) -> str | None:
    if not tag.startswith("language:"):
        return None
    return tag.split(":", 1)[1].strip().lower() or None


def _extract_languages(tags: Iterable[str]) -> list[str]:
    langs: list[str] = []
    for t in tags:
        lang = _normalize_lang_tag(t)
        if lang and lang not in langs:
            langs.append(lang)
    return langs


def _extract_license(card_data: Any) -> str | None:
    if not isinstance(card_data, dict):
        return None
    lic = card_data.get("license")
    if isinstance(lic, str) and lic.strip():
        return lic.strip()
    if isinstance(lic, list):
        # Sometimes HF stores license as list of strings
        items = [str(x).strip() for x in lic if str(x).strip()]
        return ", ".join(items) if items else None
    return None


def _extract_license_from_tags(tags: Iterable[str]) -> str | None:
    licenses: list[str] = []
    for t in tags:
        if not isinstance(t, str):
            continue
        if t.startswith("license:"):
            val = t.split(":", 1)[1].strip()
            if val and val not in licenses:
                licenses.append(val)
    return ", ".join(licenses) if licenses else None


def _matches_lang(info: DatasetInfo, wanted_langs: set[str]) -> bool:
    if not wanted_langs:
        return True
    if any(lang in wanted_langs for lang in info.languages):
        return True
    # Fallback: sometimes language tags are missing; do not exclude hard.
    return False


def list_datasets(query: str, limit: int) -> list[DatasetInfo]:
    api = HfApi()
    results = api.list_datasets(search=query, limit=limit, full=True)
    out: list[DatasetInfo] = []

    for r in results:
        tags = list(getattr(r, "tags", []) or [])
        card_data = getattr(r, "cardData", None)
        lic = _extract_license(card_data) or _extract_license_from_tags(tags)
        out.append(
            DatasetInfo(
                dataset_id=str(getattr(r, "id", "")),
                license=lic,
                languages=_extract_languages(tags),
                tags=tags,
                likes=getattr(r, "likes", None),
                downloads=getattr(r, "downloads", None),
                last_modified=str(getattr(r, "lastModified", None))
                if getattr(r, "lastModified", None) is not None
                else None,
            )
        )

    return [d for d in out if d.dataset_id]


def to_markdown(rows: list[DatasetInfo]) -> str:
    def esc(s: str | None) -> str:
        if not s:
            return ""
        return s.replace("|", "\\|")

    lines = [
        "| dataset_id | license | languages | downloads | likes |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    esc(r.dataset_id),
                    esc(r.license),
                    esc(",".join(r.languages) if r.languages else ""),
                    str(r.downloads or ""),
                    str(r.likes or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "List Hugging Face datasets matching a query, with license and language tags."
        )
    )
    p.add_argument("--query", required=True, help="Search query (e.g. 'medical', 'triage', 'mcqa')")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument(
        "--languages",
        default="fr,en",
        help=(
            "Comma-separated language tags to keep (e.g. 'fr,en'). If empty, keep all. "
            "Note: language tags may be missing on HF cards."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output file path (markdown).",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Optional output file path (json).",
    )

    args = p.parse_args()
    wanted = {x.strip().lower() for x in str(args.languages).split(",") if x.strip()}

    rows = list_datasets(args.query, limit=args.limit)
    rows = [r for r in rows if _matches_lang(r, wanted)]

    md = to_markdown(rows)
    print(md)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
