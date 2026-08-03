"""Smoke tests for the /sciencerag/validate route (spec §4.5, M2 scope).

Runs the real tec_surrogate checks (no mocking) — they're local numpy/torch
computation over small bundled fixtures, not an external API, so this stays
fast and free like the rest of the suite.
"""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.validate import tec_bridge

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "validate.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["ValidateResponse"]

client = TestClient(app)


def _report_row(index: int) -> tuple[dict[str, float], dict[str, float]]:
    dataset = tec_bridge.load_report_dataset()
    input_names = list(dataset["input_names"])
    scalar_names = list(dataset["scalar_names"])
    inverse_names = {latent: contract for contract, latent in tec_bridge.CONTRACT_TO_LATENT_INPUT.items()}
    design_parameters = {
        inverse_names[name]: float(dataset["X"][index, input_names.index(name)])
        for name in input_names
        if name in inverse_names
    }
    scalar_results = {
        name: float(dataset["scalar_outputs"][index, scalar_names.index(name)])
        for name in scalar_names
    }
    return design_parameters, scalar_results


def test_minimal_request_returns_valid_schema() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={"run_id": "run_minimal", "design_parameters": {}, "n_pairs": 1, "priors": []},
    )
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["status"] == "ok"
    assert {a["check"] for a in payload["anomalies"]} == {"energy_balance", "pde_residual", "ood"}
    # nothing to check against without field_case_index/latent_state -> info, not blocking
    assert all(a["severity"] == "info" for a in payload["anomalies"])
    assert payload["update_package"]["blocked"] is False
    assert payload["update_package"]["surrogate_update"] is None
    assert payload["update_package"]["kg_candidates"] == []


def test_matching_benchmark_case_is_consistent() -> None:
    design_parameters, scalar_results = _report_row(0)
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_benchmark_match",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": [],
        },
    )
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["evaluation"]["verdict"] == "consistent"
    assert len(payload["evaluation"]["deviations"]) == len(scalar_results)
    assert all(d["verdict"] == "within_range" for d in payload["evaluation"]["deviations"])
    # M3 (4.4): a consistent, non-blocked run with real scalar_results is
    # exactly the case spec §4.4 extracts candidate triples from.
    assert payload["update_package"]["surrogate_update"] is None  # nothing anomalous to suggest
    candidates = payload["update_package"]["kg_candidates"]
    assert len(candidates) == len(scalar_results)
    assert {c["relation"] for c in candidates} == {f"achieves_{name}" for name in scalar_results}
    assert all(c["dedup_status"] == "new" for c in candidates)
    assert all(c["confidence"] == pytest.approx(0.7) for c in candidates)


def test_benchmark_deviation_is_flagged() -> None:
    design_parameters, scalar_results = _report_row(0)
    scalar_results = dict(scalar_results)
    first_field = next(iter(scalar_results))
    scalar_results[first_field] *= 5
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_benchmark_deviation",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": [],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "deviation_found"
    flagged = [d for d in payload["evaluation"]["deviations"] if d["field"] == first_field]
    assert flagged[0]["verdict"] == "deviation"
    # M3 (4.3): the deviation should surface as an error-driven training
    # sample recommendation. (4.4): deviation_found runs are excluded from
    # KG extraction (spec §4.2 — a finding for a human, not confirmed fact).
    surrogate_update = payload["update_package"]["surrogate_update"]
    assert surrogate_update is not None
    assert any(
        sample["region"] == f"benchmark_comparison:{first_field}"
        for sample in surrogate_update["recommended_training_samples"]
    )
    assert payload["update_package"]["kg_candidates"] == []


def test_non_matching_design_is_insufficient_benchmark() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_no_match",
            "design_parameters": {"leg_length": 999.0},
            "n_pairs": 1,
            "scalar_results": {"delta_T_max_K": 50.0},
            "priors": [],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "insufficient_benchmark"


def test_prior_out_of_range_is_deviation() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_prior_deviation",
            "design_parameters": {"leg_length": 0.5},
            "n_pairs": 1,
            "priors": [
                {
                    "prior_id": "pr_test_1",
                    "kind": "parameter_range",
                    "field": "leg_length",
                    "value": {"field_name": "leg_length", "min": 5.0, "max": 10.0, "unit": "mm"},
                    "confidence": 0.8,
                    "sources": [{"type": "paper", "doi": "10.1/x"}],
                }
            ],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "deviation_found"
    deviation = payload["evaluation"]["deviations"][0]
    assert deviation["source"] == "prior_comparison"
    assert deviation["reference_id"] == "pr_test_1"
    assert deviation["verdict"] == "deviation"


def test_extreme_latent_state_blocks_update_package() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_ood_extreme",
            "design_parameters": {},
            "n_pairs": 1,
            "latent_state": [1000.0, -1000.0, 500.0, 0.0, 0.0],
            "priors": [],
        },
    )
    payload = response.json()
    ood_anomaly = next(a for a in payload["anomalies"] if a["check"] == "ood")
    assert ood_anomaly["severity"] == "blocking"
    assert payload["update_package"]["blocked"] is True
    assert payload["update_package"]["surrogate_update"] is None
    assert payload["update_package"]["kg_candidates"] == []


def test_wrong_latent_dimension_is_warning_not_silent() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_bad_latent_dim",
            "design_parameters": {},
            "n_pairs": 1,
            "latent_state": [0.0, 0.0],
            "priors": [],
        },
    )
    payload = response.json()
    ood_anomaly = next(a for a in payload["anomalies"] if a["check"] == "ood")
    assert ood_anomaly["severity"] == "warning"
    assert "error" in ood_anomaly["evidence"]
    # M3 (4.3): a warning-severity (not blocking) anomaly is exactly the
    # uncertainty-driven signal spec §4.3 selects fine-tune samples from.
    assert payload["update_package"]["blocked"] is False
    surrogate_update = payload["update_package"]["surrogate_update"]
    assert surrogate_update is not None
    assert any(
        sample["region"] == "check:ood" for sample in surrogate_update["recommended_training_samples"]
    )


def test_conservation_check_skips_multi_pair() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_multi_pair",
            "design_parameters": {},
            "n_pairs": 6,
            "field_case_index": 0,
            "priors": [],
        },
    )
    payload = response.json()
    energy_anomaly = next(a for a in payload["anomalies"] if a["check"] == "energy_balance")
    assert energy_anomaly["severity"] == "info"
    assert energy_anomaly["evidence"]["skipped"] is True
    # PDE residual, unlike conservation, does cover n_pairs > 1
    pde_anomaly = next(a for a in payload["anomalies"] if a["check"] == "pde_residual")
    assert pde_anomaly["evidence"].get("skipped") is not True
    assert pde_anomaly["evidence"]["calibration"] == "composed_topology_pending_multipair_comsol"


def test_field_case_index_out_of_range_is_rejected() -> None:
    response = client.post(
        "/sciencerag/validate",
        json={"run_id": "run_bad_index", "field_case_index": 99, "priors": []},
    )
    assert response.status_code == 422
