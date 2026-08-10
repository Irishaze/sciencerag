"""Smoke tests for the /sciencerag/report route (spec §5, M4)."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.report import store

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "report.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["ReportResponse"]

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_reports_dir(tmp_path, monkeypatch):
    """Without this, every test run writes real report files into
    data/reports/ — the same directory the live app's Reports page lists
    from — leaving hundreds of run_report_test_* fixtures behind."""
    monkeypatch.setattr(store, "REPORTS_DIR", tmp_path)

_BASE_PAYLOAD = {
    "run_id": "run_report_test",
    "task_context": {"objective": "maximize_cop", "constraints": {"heat_load_w": 5.0}},
    "design_parameters": {"leg_length": 1.0},
    "n_pairs": 1,
    "scalar_results": {"delta_T_max_K": 70.0},
    "priors": [
        {
            "prior_id": "pr_1",
            "kind": "parameter_range",
            "field": "leg_length",
            "value": {"field_name": "leg_length", "min": 0.5, "max": 2.0, "unit": "mm"},
            "confidence": 0.8,
            "sources": [{"type": "paper", "doi": "10.1/x"}],
        }
    ],
    "anomalies": [{"check": "ood", "severity": "info", "evidence": {}}],
    "evaluation": {"verdict": "consistent", "deviations": [], "sources": []},
    "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": False},
}


def test_report_returns_valid_schema_and_citations():
    response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["run_id"] == "run_report_test"
    assert payload["key_results"][0]["confidence_label"] == "high"
    assert payload["citations"] == [{"type": "paper", "doi": "10.1/x", "span": None}]
    assert "delta_T_max_K" in payload["markdown"]
    assert "10.1/x" in payload["markdown"]


def test_warning_anomaly_flags_key_results_as_check_flagged():
    payload = {**_BASE_PAYLOAD, "anomalies": [{"check": "ood", "severity": "warning", "evidence": {}}]}
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert body["key_results"][0]["confidence_label"] == "check_flagged"


def test_no_anomalies_labeled_no_anomaly_data():
    payload = {**_BASE_PAYLOAD, "anomalies": []}
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert body["key_results"][0]["confidence_label"] == "no_anomaly_data"


def test_blocked_run_summarizes_as_blocked_no_updates():
    payload = {
        **_BASE_PAYLOAD,
        "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": True},
    }
    response = client.post("/sciencerag/report", json=payload)
    body = response.json()
    assert "blocked" in body["markdown"].lower()


def test_report_is_persisted_and_listable():
    response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    trace_id = response.json()["trace_id"]
    entries = store.list_reports()
    matching = [e for e in entries if e["stem"].startswith("run_report_test_")]
    assert matching
    loaded = store.load_report(matching[0]["stem"])
    assert loaded.run_id == "run_report_test"


def test_reports_listing_and_fetch_endpoints():
    post_response = client.post("/sciencerag/report", json=_BASE_PAYLOAD)
    trace_id = post_response.json()["trace_id"]

    list_response = client.get("/sciencerag/reports")
    assert list_response.status_code == 200
    stems = [entry["stem"] for entry in list_response.json()]
    matching = [stem for stem in stems if stem.startswith("run_report_test_")]
    assert matching

    fetch_response = client.get(f"/sciencerag/reports/{matching[0]}")
    assert fetch_response.status_code == 200
    assert fetch_response.json()["trace_id"] == trace_id


def test_fetch_nonexistent_report_is_404():
    response = client.get("/sciencerag/reports/does_not_exist")
    assert response.status_code == 404


@pytest.mark.parametrize("bad_run_id", ["../../etc/passwd", "a/b", "a\\b"])
def test_run_id_with_path_separator_is_rejected(bad_run_id: str, tmp_path):
    """Adversarial test: run_id flows unsanitized into report/store.py's
    filename construction. Confirmed for real: run_id="../kg/marker" made
    POST /sciencerag/report write a file onto the HOST filesystem outside
    data/reports/ entirely (data/ is bind-mounted in docker-compose.yml,
    so this reaches real host paths, not just the container's). Reject at
    the API boundary."""
    payload = {**_BASE_PAYLOAD, "run_id": bad_run_id}
    response = client.post("/sciencerag/report", json=payload)
    assert response.status_code == 422
    # And nothing should have been written anywhere, including outside tmp_path.
    assert list(tmp_path.iterdir()) == []


def test_non_finite_scalar_result_is_rejected():
    """Adversarial test: a NaN scalar_result was confirmed to flow through
    unblocked, get embedded in update_package.kg_candidates, and get
    written into data/kg/graph.json as a literal non-standard NaN JSON
    token once approved via scripts/approve_kg_candidates.py."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {**_BASE_PAYLOAD, "scalar_results": {"delta_T_max_K": bad}}
        response = client.post(
            "/sciencerag/report",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, f"scalar_results={bad} should be rejected"
