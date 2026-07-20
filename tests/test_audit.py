"""Tests for the JSONL audit log (sciencerag/common/audit.py) and its wiring
into the /sciencerag/priors route."""

import json

from fastapi.testclient import TestClient

from sciencerag.common import audit
from sciencerag.common.audit import log_audit_entry
from sciencerag.priors import router as priors_router
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper


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

    def fake_build_priors_response(query: str) -> PriorsResponse:
        return PriorsResponse(
            priors=[
                Prior(
                    prior_id="pr_fake_0001",
                    kind="parameter_range",
                    field="general_finding",
                    value={"summary": f"fake evidence for: {query}"},
                    confidence=0.8,
                    sources=[SourcePaper(doi="10.0000/fake", span="pages 1-2")],
                )
            ],
            coverage=Coverage(internal_hits=1, external_hits=0, gaps=[]),
            trace_id="tr_fake_audit_test",
        )

    monkeypatch.setattr(
        priors_router.retrieval, "build_priors_response", fake_build_priors_response
    )

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
