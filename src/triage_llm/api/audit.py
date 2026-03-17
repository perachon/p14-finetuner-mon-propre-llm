from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triage_llm.utils import ensure_dir


class AuditStore:
    def __init__(self, db_path: str = "runs/audit/audit.db") -> None:
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )

    def save(self, request: dict[str, Any], response: dict[str, Any]) -> str:
        interaction_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                (
                    "INSERT INTO interactions (id, created_at, request_json, response_json) "
                    "VALUES (?, ?, ?, ?)"
                ),
                (
                    interaction_id,
                    created_at,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                ),
            )
        return interaction_id

    def get(self, interaction_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, created_at, request_json, response_json FROM interactions WHERE id = ?",
                (interaction_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "request": json.loads(row[2]),
            "response": json.loads(row[3]),
        }
