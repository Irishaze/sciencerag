"""Tests for the /sciencerag/kg_candidates/pending web-panel routes —
sciencerag/kg_approval/router.py, the HTTP equivalent of scripts/
approve_kg_candidates.py's --pending queue flow. Both share the same
underlying logic (sciencerag/validate/kg_approval.py), so behavior here
must match test_approve_kg_candidates_script.py's CLI-level assertions."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.common import audit
from sciencerag.priors import kg
from sciencerag.validate import kg_candidate_store
from sciencerag.validate.models import KGCandidate

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "kg_approval.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")


def _queue_candidates(run_id="run_test", n=1) -> str:
    candidates = [
        KGCandidate.model_validate(
            {
                "subject": "Bi2Te3 single-stage TEC",
                "relation": f"achieves_delta_T_max_K_{i}",
                "object_value": 70.0 + i,
                "object_unit": "K",
                "conditions": {},
                "confidence": 0.7,
                "run_id": run_id,
                "dedup_status": "new",
                "entity_type": "TECDesign",
            }
        )
        for i in range(n)
    ]
    kg_candidate_store.store_pending_candidates(run_id, candidates)
    return kg_candidate_store.list_pending()[0]["stem"]


def test_list_pending_empty():
    response = client.get("/sciencerag/kg_candidates/pending")
    assert response.status_code == 200
    assert response.json() == []


def test_list_pending_reports_queued_batch():
    stem = _queue_candidates(n=2)
    response = client.get("/sciencerag/kg_candidates/pending")
    assert response.status_code == 200
    jsonschema.validate(instance=response.json(), schema={"type": "array", "items": SCHEMA["PendingBatchSummary"]})
    assert response.json() == [{"stem": stem, "count": 2}]


def test_get_pending_returns_full_candidates():
    stem = _queue_candidates(n=2)
    response = client.get(f"/sciencerag/kg_candidates/pending/{stem}")
    assert response.status_code == 200
    jsonschema.validate(instance=response.json(), schema=SCHEMA["PendingBatchDetail"])
    assert response.json()["stem"] == stem
    assert len(response.json()["candidates"]) == 2


def test_get_pending_missing_stem_is_404():
    response = client.get("/sciencerag/kg_candidates/pending/does_not_exist")
    assert response.status_code == 404
    assert response.json()["status"] == "error"


def test_get_pending_path_traversal_stem_is_rejected():
    response = client.get("/sciencerag/kg_candidates/pending/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (400, 404)  # never 200, never leaks outside PENDING_DIR


def test_approve_all_writes_triples_and_archives():
    stem = _queue_candidates(n=2)
    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"approve_all": True, "operator": "tester", "reason": "unit test"},
    )
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=SCHEMA["ApproveResponse"])
    assert payload["archived"] is True
    assert len(payload["results"]) == 2
    assert all(r["status"] == "added" for r in payload["results"])
    assert all(r["triple_id"] for r in payload["results"])

    # Same archive behavior the CLI has — no longer pending, archived not deleted.
    assert kg_candidate_store.list_pending() == []
    assert (kg_candidate_store.ARCHIVE_DIR / f"{stem}.json").exists()

    triples = kg._load_triples()
    assert len(triples) == 2


def test_approve_carries_evidence_detail_into_the_final_triple():
    # Regression for the 2026-08-14/15 "方案B" wiring: supporting_evidence's
    # deviation_detail (the concrete numbers behind confidence — real
    # relative deviation + which benchmark case) used to be silently
    # dropped at approval, since add_triple() had no field to receive it.
    candidate = KGCandidate.model_validate(
        {
            "subject": "Bi2Te3 single-stage TEC",
            "relation": "achieves_delta_T_max_K",
            "object_value": 71.7,
            "object_unit": "K",
            "conditions": {},
            "confidence": 0.85,
            "run_id": "run_evidence_detail",
            "dedup_status": "new",
            "entity_type": "TECDesign",
            "supporting_evidence": {
                "deviation_detail": {
                    "verdict": "consistent",
                    "relative_deviation": 0.008,
                    "benchmark_case_id": "sample_02.docx",
                }
            },
        }
    )
    kg_candidate_store.store_pending_candidates("run_evidence_detail", [candidate])
    stem = kg_candidate_store.list_pending()[0]["stem"]

    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"approve_all": True, "operator": "tester", "reason": "unit test"},
    )
    assert response.status_code == 200

    triples = kg._load_triples()
    assert len(triples) == 1
    assert triples[0].evidence_detail == {
        "verdict": "consistent",
        "relative_deviation": 0.008,
        "benchmark_case_id": "sample_02.docx",
    }


def test_approve_specific_indices_only():
    stem = _queue_candidates(n=3)
    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"indices": [0, 2], "operator": "tester", "reason": ""},
    )
    payload = response.json()
    assert len(payload["results"]) == 2
    assert {r["index"] for r in payload["results"]} == {0, 2}
    # Still archives the whole batch, same as a CLI partial --approve does
    # — index 1 isn't lost, just no longer in the default pending queue.
    assert kg_candidate_store.list_pending() == []
    assert len(kg._load_triples()) == 2


def test_approve_out_of_range_index_reports_error_not_crash():
    stem = _queue_candidates(n=1)
    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"indices": [5], "operator": "tester", "reason": ""},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == [{"index": 5, "status": "error", "triple_id": None, "error": "index out of range"}]
    assert kg._load_triples() == []


def test_approve_missing_selection_is_400():
    stem = _queue_candidates(n=1)
    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"operator": "tester", "reason": ""},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "error"
    # Nothing should have been touched.
    assert len(kg_candidate_store.list_pending()) == 1


def test_approve_missing_stem_is_404():
    response = client.post(
        "/sciencerag/kg_candidates/pending/does_not_exist/approve",
        json={"approve_all": True},
    )
    assert response.status_code == 404


def test_null_byte_in_stem_is_rejected_not_crashed():
    """Adversarial test: a stem containing a null byte passed
    reject_path_unsafe_id's old check (it only blocked "/"/"\\") and reached
    pathlib inside load_pending, which raises ValueError("embedded null
    byte") — uncaught, surfacing as a raw 500 instead of the router's typed
    400/404. Reject at the same boundary "/"/"\\" already are."""
    get_response = client.get("/sciencerag/kg_candidates/pending/a%00b")
    assert get_response.status_code == 400
    assert get_response.json()["status"] == "error"

    post_response = client.post(
        "/sciencerag/kg_candidates/pending/a%00b/approve", json={"approve_all": True}
    )
    assert post_response.status_code == 400


def test_crlf_and_quote_in_stem_is_rejected_not_crashed():
    """Adversarial test (2026-08-15 full-system review): a stem containing
    '"'/'\\r'/'\\n' passed the old check the same way the null byte did —
    confirmed via a real end-to-end repro on sciencerag.report (not this
    router, but the same shared reject_path_unsafe_id): a run_id of
    'inject"x\\r\\nX-Injected-Header: pwned' got written into two real
    filenames on disk verbatim, and fetching that report's /pdf endpoint
    crashed the connection (h11 raising LocalProtocolError on the resulting
    malformed Content-Disposition header, deep past this project's own
    try/except) instead of a clean 4xx. Same boundary, same fix."""
    for bad_stem in ('a%22b', 'a%0Db', 'a%0Ab'):  # ", \r, \n
        get_response = client.get(f"/sciencerag/kg_candidates/pending/{bad_stem}")
        assert get_response.status_code == 400, bad_stem
        assert get_response.json()["status"] == "error"


def test_approve_indices_over_max_length_is_rejected():
    """Adversarial test: an unbounded indices list let a single request with
    5,000,000 out-of-range indices burn ~9s of server CPU and produce a
    409MB response body, from a request body a few hundred KB — attacker
    cost negligible, server cost large. A real pending batch never comes
    close to 1000 candidates."""
    stem = _queue_candidates(n=1)
    response = client.post(
        f"/sciencerag/kg_candidates/pending/{stem}/approve",
        json={"indices": list(range(1001))},
    )
    assert response.status_code == 422
