from __future__ import annotations

from fastapi.testclient import TestClient

from triage_llm.api.app import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
