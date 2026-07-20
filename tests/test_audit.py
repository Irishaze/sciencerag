"""Tests for the JSONL audit log (sciencerag/common/audit.py) and its wiring
into the /sciencerag/priors route."""

import json

from fastapi.testclient import TestClient

from sciencerag.common import audit
from sciencerag.common.audit import log_audit_entry


def test_log_audit_entry_appends_one_line_per_call(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")

    log_audit_entry(
        trace_id="tr_1",
        endpoint="sciencerag.priors",
        request={"query": "a"},
        evidence=[],
        output={"status": "ok"},
    )
    log_audit_entry(
        trace_id="tr_2",
        endpoint="sciencerag.priors",
        request={"query": "b"},
        evidence=[],
        output={"status": "ok"},
    )

    lines = audit.AUDIT_LOG_PATH.read_text().splitlines()
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["trace_id"] == "tr_1"
    assert entry1["endpoint"] == "sciencerag.priors"
    assert entry1["request"] == {"query": "a"}
    assert "timestamp" in entry1


def test_priors_route_writes_audit_log_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")

    from sciencerag.app import app

    client = TestClient(app)
    resp = client.post("/sciencerag/priors", json={"query": "audit log smoke test"})
    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]

    lines = audit.AUDIT_LOG_PATH.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["trace_id"] == trace_id
    assert entry["request"]["query"] == "audit log smoke test"
    assert entry["output"]["trace_id"] == trace_id
