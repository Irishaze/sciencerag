"""M1-20: audit log completeness (spec §3.5/§8) — given any trace_id, the
JSONL audit log must hold enough to reconstruct the full request, the
evidence actually cited, and the output returned to the caller, for BOTH
the success and error paths. Mocks retrieval so this stays fast/free.
"""

import json

from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.common import audit
from sciencerag.priors import router as priors_router
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper

client = TestClient(app)


def _read_last_entry() -> dict:
    lines = audit.AUDIT_LOG_PATH.read_text().splitlines()
    return json.loads(lines[-1])


def _fake_priors_response(query: str, allow_external: bool = False) -> tuple[PriorsResponse, int]:
    return (
        PriorsResponse(
            priors=[
                Prior(
                    prior_id="pr_fake_0001",
                    kind="material_property",
                    field="seebeck_coefficient",
                    value={"typical": 200, "unit": "uV/K"},
                    confidence=0.75,
                    sources=[
                        SourcePaper(doi="10.1111/aaa", span="p.3"),
                        SourcePaper(doi="10.2222/bbb", span="p.5"),
                    ],
                    notes="test note",
                )
            ],
            coverage=Coverage(
                internal_hits=4, external_hits=0, gaps=["1 low-confidence prior excluded"]
            ),
            trace_id="tr_completeness_test",
        ),
        1,
    )


def test_success_entry_fully_reconstructs_request_evidence_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _fake_priors_response)

    payload = {
        "query": "what is the seebeck coefficient",
        "task_context": {
            "objective": "design a cooler",
            "constraints": {"heat_load_w": 5.0},
        },
        "max_priors": 3,
        "allow_external": False,
    }
    resp = client.post("/sciencerag/priors", json=payload)
    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]

    entry = _read_last_entry()
    assert entry["trace_id"] == trace_id

    # 1. Request reconstructable in full, including nested task_context.
    assert entry["request"] == payload

    # 2. Evidence must reflect every source actually cited by the returned
    #    priors, not an empty placeholder.
    cited_dois = {s["doi"] for prior in resp.json()["priors"] for s in prior["sources"]}
    logged_dois = {e["doi"] for e in entry["evidence"]}
    assert logged_dois == cited_dois == {"10.1111/aaa", "10.2222/bbb"}

    # 3. Output must match what the caller actually received.
    assert entry["output"] == resp.json()

    # 4. Model/embedding config and elapsed time traceable per spec §3.5/§9.
    assert entry["model_config"]["llm_model"]
    assert entry["model_config"]["embedding_model"]
    assert entry["elapsed_s"] is not None
    assert "timestamp" in entry
    assert entry["endpoint"] == "sciencerag.priors"

    # 5. Filtered material_property drafts (spec §3.6) are auditable even
    #    though they never appear in coverage.gaps or the response body.
    assert entry["filtered_material_count"] == 1


def test_error_entry_still_reconstructs_request_and_trace_id(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")

    def _boom(query: str, allow_external: bool = False) -> PriorsResponse:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _boom)

    payload = {"query": "will fail"}
    resp = client.post("/sciencerag/priors", json=payload)
    assert resp.status_code == 502
    trace_id = resp.json()["trace_id"]

    entry = _read_last_entry()
    assert entry["trace_id"] == trace_id
    assert entry["request"]["query"] == "will fail"
    assert entry["output"]["status"] == "error"
    assert entry["output"]["error"]["message"] == "simulated failure"
    assert entry["evidence"] == []
    assert entry["elapsed_s"] is not None
