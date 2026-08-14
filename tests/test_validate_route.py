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
from sciencerag.priors import kg
from sciencerag.validate import kg_candidate_store, kg_candidates as kg_candidates_module, tec_bridge
from sciencerag.validate.checks import BLOCKING_MARGIN

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "validate.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["ValidateResponse"]

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_graph(tmp_path, monkeypatch):
    """4.4's KG candidate extraction calls kg.query_kg() for dedup_status
    (sciencerag/validate/kg_candidates.py) — without this, these tests
    depend on the real data/kg/graph.json happening to be empty, which
    isn't guaranteed once anything else (scripts/demo_end_to_end.py, a
    real approval run) has ever written to it. Same isolation test_ask_route.py
    and test_kg_graph.py already use."""
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")
    # Same reasoning for the pending-candidates queue: without this, every
    # run here would drop a real file into data/kg_candidates/pending/.
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "kg_candidates_pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "kg_candidates_archive")
    # Regression found 2026-08-11: once a real data/kg/ontology.json
    # existed on disk, extract_kg_candidates() started making real
    # classify_relation/describe_relation LLM calls on every test run here
    # — tests must not depend on incidental host filesystem state.
    monkeypatch.setattr(kg_candidates_module, "load_ontology", lambda: None)


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
    assert {a["check"] for a in payload["anomalies"]} == {"ood"}
    # nothing to check against without latent_state -> info, not blocking
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
    # len(scalar_results) numeric fact candidates + 2 structural link
    # candidates (SimulationRun -> TECDesign, SimulationRun -> Material) —
    # see kg_candidates.py's own comment on why those two always get
    # appended alongside the measured-value ones.
    assert len(candidates) == len(scalar_results) + 2
    numeric_candidates = [c for c in candidates if c["object_value"] is not None]
    link_candidates = [c for c in candidates if c["object_entity_id"] is not None]
    assert {c["relation"] for c in numeric_candidates} == {f"achieves_{name}" for name in scalar_results}
    assert {c["relation"] for c in link_candidates} == {"SIMULATION_USES_DESIGN", "SIMULATION_USES_MATERIAL"}
    assert all(c["dedup_status"] == "new" for c in candidates)
    # _report_row echoes a benchmark case's scalar_results back exactly, so
    # relative_deviation=0 for every field — the top of the per-field
    # confidence range (2026-08-14/15: _consistent_confidence, no longer a
    # flat 0.7 regardless of how close the match actually was).
    assert all(c["confidence"] == pytest.approx(0.9) for c in numeric_candidates)
    assert all(c["confidence"] == pytest.approx(1.0) for c in link_candidates)
    # Non-empty kg_candidates should be queued for approve_kg_candidates.py
    # --list-pending, not just returned in the response body.
    pending = kg_candidate_store.list_pending()
    assert any(entry["stem"].startswith("run_benchmark_match_") for entry in pending)
    matching_entry = next(e for e in pending if e["stem"].startswith("run_benchmark_match_"))
    assert matching_entry["count"] == len(candidates)


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


def test_wide_tolerance_field_within_it_is_consistent_not_flagged() -> None:
    # Regression for the 2026-08-14/15 finding (scripts/loo_scalar_error_sweep.py):
    # total_resistance_ohm has real leave-one-out relative error up to ~90th
    # percentile 219% against the deployed surrogate — the old flat 5%
    # BENCHMARK_SCALAR_RELATIVE_TOLERANCE made "consistent" essentially
    # unreachable for this field from a genuine prediction (confirmed
    # separately: every historical "consistent" verdict in logs/audit.jsonl
    # had exactly 0 deviation, i.e. an echoed exact value, not a real
    # prediction). A 30% deviation on total_resistance_ohm — which the old
    # flat 5% band would have flagged as deviation_found — must now land
    # within its widened per-field tolerance (0.5) instead.
    design_parameters, scalar_results = _report_row(0)
    scalar_results = dict(scalar_results)
    scalar_results["total_resistance_ohm"] *= 1.30
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_wide_tolerance",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": [],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "consistent"
    flagged = [d for d in payload["evaluation"]["deviations"] if d["field"] == "total_resistance_ohm"]
    assert flagged[0]["verdict"] == "within_range"


def test_tight_tolerance_field_still_flags_small_deviation() -> None:
    # The other half of the same finding: delta_T_max_K/figure_of_merit_1_per_K
    # generalize tightly (real leave-one-out p90 <1.5%) and correctly kept
    # their existing 5% tolerance, unchanged and unwidened — a 10% deviation
    # there must still flag as a real deviation, not get waved through by
    # whatever widening applies to the unreliable fields.
    design_parameters, scalar_results = _report_row(0)
    scalar_results = dict(scalar_results)
    scalar_results["delta_T_max_K"] *= 1.10
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_tight_tolerance",
            "design_parameters": design_parameters,
            "n_pairs": 1,
            "scalar_results": scalar_results,
            "priors": [],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "deviation_found"
    flagged = [d for d in payload["evaluation"]["deviations"] if d["field"] == "delta_T_max_K"]
    assert flagged[0]["verdict"] == "deviation"


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


def test_prior_only_consistent_match_keeps_flat_confidence_by_design() -> None:
    """Documents a real constraint, not a bug: a "consistent" verdict
    reached purely via a parameter_range prior match (no matching benchmark
    case at all) still gets the flat 0.7 confidence / no deviation_detail
    for its achieves_* candidates. This looks at first glance like the same
    "two very different matches treated identically" gap the 2026-08-14/15
    graduated-confidence fix solved for the benchmark path — it isn't:
    checked live whether widening kg_candidates.py's matching to include
    prior_comparison deviations would help, and it structurally can't.
    evaluation.py's prior comparison only ever grades a request.
    design_parameters field (e.g. "leg_length"); the achieves_* candidate
    being scored here is a request.scalar_results field (e.g.
    "delta_T_max_K") — GEOMETRY_FREE_NAMES and tec_bridge.SCALAR_UNITS'
    keys are disjoint by construction, so there is no per-field comparison
    data connecting a geometry-parameter prior to a performance-result
    candidate. The flat fallback here is the honest answer, not a gap."""
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_prior_only_consistent",
            "design_parameters": {"leg_length": 1.5},
            "n_pairs": 1,
            "scalar_results": {"delta_T_max_K": 60.0},
            "priors": [
                {
                    "prior_id": "pr_prior_only",
                    "kind": "parameter_range",
                    "field": "leg_length",
                    "value": {"field_name": "leg_length", "min": 1.0, "max": 2.0, "unit": "mm"},
                    "confidence": 0.6,
                    "sources": [{"type": "paper", "doi": "10.1/x"}],
                }
            ],
        },
    )
    payload = response.json()
    assert payload["evaluation"]["verdict"] == "consistent"
    numeric = next(c for c in payload["update_package"]["kg_candidates"] if c["object_value"] is not None)
    assert numeric["confidence"] == 0.7
    assert "deviation_detail" not in numeric["supporting_evidence"]


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
    # Regression test: a dimension mismatch means no OOD score was ever
    # computed and there's no valid latent coordinate to speak of — confirmed
    # live that suggest_surrogate_update previously claimed the generic "OOD
    # score in the upper tail... near this design point" direction anyway,
    # which is a specific, false statistical claim for a run that literally
    # couldn't be scored. It must not appear here.
    assert "upper tail" not in surrogate_update["hyperparameter_direction"]
    assert "design point" not in surrogate_update["hyperparameter_direction"]


def test_high_n_pairs_is_accepted_and_carried_into_kg_conditions() -> None:
    """n_pairs is allowed up to 500 (real commercial multi-pair modules,
    e.g. the "127" in a part number like TEC1-12706) — it no longer feeds
    any physics check (energy_balance/pde_residual were removed), only
    benchmark/prior comparison and KG candidate `conditions`."""
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_127_pairs",
            "design_parameters": {},
            "n_pairs": 127,
            "scalar_results": {"delta_T_max_K": 50.0},
            "priors": [],
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert {a["check"] for a in payload["anomalies"]} == {"ood"}
    numeric = next(c for c in payload["update_package"]["kg_candidates"] if c["object_value"] is not None)
    assert numeric["conditions"]["n_pairs"] == 127.0


@pytest.mark.parametrize("bad_run_id", ["../../etc/passwd", "a/b", "a\\b"])
def test_run_id_with_path_separator_is_rejected(bad_run_id: str) -> None:
    """Adversarial test: run_id flows unsanitized into a filename in both
    report/store.py and validate/kg_candidate_store.py — confirmed a
    crafted run_id like "../kg/marker" made POST /sciencerag/report write
    a file onto the host filesystem outside data/reports/ entirely (data/
    is bind-mounted in docker-compose.yml). Reject at the API boundary."""
    response = client.post(
        "/sciencerag/validate",
        json={"run_id": bad_run_id, "design_parameters": {}, "n_pairs": 1, "priors": []},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["design_parameters", "scalar_results"])
def test_non_finite_numeric_values_are_rejected(field: str) -> None:
    """Adversarial test: NaN/Infinity are valid Python floats, so Pydantic
    accepts them by default. A NaN scalar_result was confirmed to flow
    unblocked into a KG candidate, get auto-queued for approval, and get
    written into data/kg/graph.json as a literal non-standard NaN JSON
    token by scripts/approve_kg_candidates.py. Reject at the API boundary
    instead of letting a non-finite number reach permanent storage."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {
            "run_id": "run_non_finite",
            "design_parameters": {},
            "n_pairs": 1,
            "priors": [],
            field: {"leg_length": bad},
        }
        # httpx's own JSON encoder refuses to even serialize a NaN/Infinity
        # float client-side (unlike the real curl-based attack, which sent
        # raw bytes with no client-side validation at all) — build the
        # request body with stdlib json.dumps (allow_nan=True by default)
        # to reproduce the real wire format instead.
        response = client.post(
            "/sciencerag/validate",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, f"{field}={bad} should be rejected"


def test_non_finite_latent_state_is_rejected() -> None:
    """Adversarial test: without this validator, a NaN in latent_state
    didn't crash or produce invalid JSON — it silently produced a *wrong,
    confident* answer. check_ood's np.searchsorted treats NaN as larger
    than every real training score, so the request came back as
    severity="blocking", training_distance_percentile=100.0 —
    indistinguishable from a genuinely, validly extreme design."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {
            "run_id": "run_non_finite_latent",
            "design_parameters": {},
            "n_pairs": 1,
            "priors": [],
            "latent_state": [0.0, bad, 0.0, 0.0, 0.0],
        }
        response = client.post(
            "/sciencerag/validate",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, f"latent_state containing {bad} should be rejected"


def test_kg_candidate_queue_write_failure_does_not_sink_the_response(monkeypatch) -> None:
    """Adversarial test, confirmed live before the fix: store_pending_
    candidates is a best-effort side-storage convenience (spec §6.3's
    approval queue), not part of the contract the caller is waiting on.
    A disk failure there (e.g. an unwritable data/ mount) used to turn an
    otherwise fully-computed, correct response into a 502 via the router's
    blanket except, discarding real anomaly/evaluation results over a
    queue file nobody had asked for synchronously."""

    def _boom(run_id: str, candidates: list) -> None:
        raise OSError("simulated unwritable data/ mount")

    monkeypatch.setattr(
        "sciencerag.validate.router.store_pending_candidates", _boom
    )
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_queue_write_fails",
            "design_parameters": {},
            "n_pairs": 1,
            "scalar_results": {"delta_T_max_K": 50.0},
            "priors": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["update_package"]["blocked"] is False


def test_ood_blocking_uses_margin_over_historical_max_not_bare_percentile() -> None:
    """Regression test for the OOD blocking-threshold redesign (2026-08-13).
    A bare 99th-percentile-of-31-samples rule was confirmed to have no real
    statistical grounding at this sample size: resolving a genuine 99th
    percentile needs ~100 samples for even one expected point beyond it, and
    the parametric alternative (chi-square, df=5, the distribution this
    statistic's math actually assumes) already misfires on 6.5% of the real
    31 training points at its 99th-percentile cutoff -- 6x the intended 1%
    rate, because the real latent distribution has a fatter tail than a
    Gaussian model predicts. Replaced with an honest, sample-size-appropriate
    rule (see checks.py's BLOCKING_MARGIN docstring): blocking now fires only
    when a point is more than BLOCKING_MARGIN x more extreme than the single
    most extreme point ever actually validated. This test locks in that
    exact boundary rather than the old percentile-based one."""
    training_z = tec_bridge.load_training_latent()
    mean = training_z.mean(axis=0)
    std = training_z.std(axis=0, ddof=1)

    def latent_at_distance(distance: float) -> list[float]:
        # Move purely along axis 0 so every other axis sits exactly at the
        # mean (contributes 0) -- makes the resulting mahalanobis distance
        # exactly `distance` by construction, regardless of the real
        # dataset's actual mean/std values.
        z = mean.copy()
        z[0] += distance * std[0]
        return z.tolist()

    def severity_at(distance: float, run_id: str) -> str:
        response = client.post(
            "/sciencerag/validate",
            json={
                "run_id": run_id,
                "design_parameters": {},
                "n_pairs": 1,
                "latent_state": latent_at_distance(distance),
                "priors": [],
            },
        )
        anomaly = response.json()["anomalies"][0]
        return anomaly["severity"], anomaly["evidence"]

    _, evidence = severity_at(0.0, "run_boundary_probe")
    training_max = evidence["training_distance_range"][1]
    threshold = training_max * BLOCKING_MARGIN
    assert evidence["blocking_threshold"] == pytest.approx(threshold)

    just_below, _ = severity_at(threshold - 0.5, "run_just_below_threshold")
    just_above, _ = severity_at(threshold + 0.5, "run_just_above_threshold")
    assert just_below != "blocking"
    assert just_above == "blocking"
