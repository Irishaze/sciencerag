"""Smoke tests for the /sciencerag/report route (spec §5, M4)."""

import json
from pathlib import Path

import jsonschema
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.report import store

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "report.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["ReportResponse"]

client = TestClient(app)

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
    assert loaded.trace_id == trace_id
