"""Smoke tests for the /sciencerag/validate route (spec §4.5, M2 scope).

Runs the real tec_surrogate checks (no mocking) — they're local numpy/torch
computation over small bundled fixtures, not an external API, so this stays
fast and free like the rest of the suite.
"""

import json
from pathlib import Path

import jsonschema
import numpy as np
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


def test_prior_with_infinite_bounds_is_rejected() -> None:
    """Adversarial test, confirmed live before the fix: a parameter_range
    prior's min/max weren't covered by reject_non_finite_values/_list (those
    only guard ValidateRequest's own dict/list fields) even though priors
    are passed in-band by the caller, not resolved from a trusted store.
    min=-inf/max=inf made evaluation.py's `actual < min`/`actual > max`
    always False, so a wildly out-of-range design_parameters value (e.g.
    leg_length=99999.0 against real leg lengths of ~0.02-0.2mm) came back
    verdict="within_range", evaluation.verdict="consistent", HTTP 200 — a
    forged clean bill of health with no deviation ever raised."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        payload = {
            "run_id": "run_infinite_prior_bounds",
            "design_parameters": {"leg_length": 99999.0},
            "n_pairs": 1,
            "priors": [
                {
                    "prior_id": "pr_evil",
                    "kind": "parameter_range",
                    "field": "leg_length",
                    "value": {"field_name": "leg_length", "min": bad, "unit": "mm"},
                    "confidence": 0.9,
                    "sources": [{"type": "paper", "doi": "10.1234/fake"}],
                }
            ],
        }
        response = client.post(
            "/sciencerag/validate",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, f"prior min={bad} should be rejected"


def test_unrecognized_scalar_result_field_produces_no_kg_candidate() -> None:
    """Adversarial test, confirmed live before the fix: scalar_results has no
    fixed vocabulary at the schema level. extract_kg_candidates() iterated
    every key in request.scalar_results with no check against
    tec_bridge.SCALAR_NAMES, so an arbitrary caller-chosen field name became
    a real "achieves_<field>" KGCandidate — auto-classified and auto-queued
    to the human-approval pending directory — even though evaluation.py's
    own benchmark comparison silently skips exactly these unrecognized
    fields (`if field not in scalar_names: continue`). Fixed to match that
    existing precedent instead of fabricating a candidate."""
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_fake_scalar_field",
            "design_parameters": {},
            "n_pairs": 1,
            "priors": [],
            "scalar_results": {"totally_made_up_field_xyz": 42.0},
        },
    )
    assert response.status_code == 200
    relations = {c["relation"] for c in response.json()["update_package"]["kg_candidates"]}
    assert not any(r.startswith("achieves_") for r in relations)


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


def test_kg_relation_classification_failure_does_not_sink_the_response(monkeypatch) -> None:
    """Adversarial test, confirmed live before the fix: classify_relation/
    describe_relation are real synchronous litellm.completion calls
    (timeout=90s, longer alone than this endpoint's own 60s
    LATENCY_TARGET_SECONDS) whenever a relation isn't already cached —
    reachable in production the moment data/kg/ontology.json exists, per
    this file's own autouse fixture monkeypatching load_ontology to None
    specifically to avoid triggering them in every other test here. Before
    the fix, a network failure from either call was uncaught, propagated
    through post_validate's blanket except, and turned an otherwise fully-
    computed, correct anomalies/evaluation result into a 502 — the same
    "best-effort side-effect must not sink the main response" failure
    already fixed for store_pending_candidates just below this function's
    own call site."""
    monkeypatch.setattr(kg_candidates_module, "load_ontology", lambda: object())

    def _boom(relation: str, ontology) -> str:
        raise TimeoutError("simulated LLM network timeout")

    monkeypatch.setattr(kg_candidates_module, "classify_relation", _boom)
    response = client.post(
        "/sciencerag/validate",
        json={
            "run_id": "run_llm_classification_fails",
            "design_parameters": {},
            "n_pairs": 1,
            "scalar_results": {"delta_T_max_K": 50.0},
            "priors": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    candidates = payload["update_package"]["kg_candidates"]
    assert any(c["relation"] == "achieves_delta_T_max_K" for c in candidates)


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


def test_ood_warning_uses_the_data_own_gap_not_a_historical_point_or_rank() -> None:
    """Regression test for the OOD warning-threshold redesign (2026-08-17,
    same first-principles pass that dropped BLOCKING_MARGIN to 1.0 — see
    that constant's docstring in checks.py), revised a second time the same
    day: an earlier version of this fix anchored warning directly to the
    2nd-highest historical score's own value (~4.49) — better than
    percentile>=90-by-rank (which landed inside the smooth, structureless
    part of the distribution), but still the same shape of mistake
    BLOCKING_MARGIN's docstring already describes: it put the boundary
    exactly on a single real historical point again, so a near-duplicate of
    that one training design would sit right on the edge. warning now fires
    past the *midpoint* of the data's own largest gap (excluding the top
    point, which blocking owns) instead — anywhere strictly inside that gap
    classifies the training data identically, so the midpoint is the
    natural, symmetric choice rather than sitting on top of either
    endpoint. This test locks in that exact boundary."""
    training_z = tec_bridge.load_training_latent()
    mean = training_z.mean(axis=0)
    std = training_z.std(axis=0, ddof=1)

    def latent_at_distance(distance: float) -> list[float]:
        z = mean.copy()
        z[0] += distance * std[0]
        return z.tolist()

    def severity_at(distance: float, run_id: str) -> tuple[str, dict]:
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

    _, evidence = severity_at(0.0, "run_warning_boundary_probe")
    threshold = evidence["warning_threshold"]
    assert threshold == pytest.approx(4.16, abs=0.02)

    just_below, _ = severity_at(threshold - 0.2, "run_just_below_warning_threshold")
    just_above, _ = severity_at(threshold + 0.2, "run_just_above_warning_threshold")
    assert just_below == "info"
    assert just_above == "warning"


def test_ood_threshold_values_are_pinned_to_todays_training_data() -> None:
    """Snapshot/pin test, not a formula-correctness test (see the test just
    above for that) — this one is meant to FAIL the moment the 31-sample
    training set changes, on purpose.

    checks.py/BLOCKING_MARGIN's own docstring already says the OOD threshold
    is provisional at n=31 and should be revisited once training data grows
    past ~100-150 samples. The risk isn't that nobody knows that — it's that
    "revisit later" living only in a docstring/delivery-doc sentence depends
    on a human remembering to go re-read it *at the moment the training set
    actually changes*, months from now, by someone who may not be the person
    who wrote that sentence.

    History (2026-08-17, same day, three revisions): this test originally
    pinned the *original* formula's output (BLOCKING_MARGIN=1.5 ->
    blocking≈19.98, percentile>=90-by-rank -> warning≈3.44). A first-
    principles re-check of those two specific numbers, not just of "formula
    vs. hardcoded", found real problems with both: BLOCKING_MARGIN's
    ">1.0 headroom" never actually protected against "one n=1 sample
    defines the boundary" the way its own docstring claimed, and extending
    trust 50% past the worst point ever validated cuts against what an OOD
    gate is for. And percentile>=90's 28th-of-31-by-rank cutoff (~3.44)
    turned out to land inside the smooth, structureless part of the real
    sorted score distribution — not at any gap the data itself shows.
    checks.py was changed to BLOCKING_MARGIN=1.0 (blocking = training_max,
    no margin) and warning = "> the 2nd-highest historical score" (~4.49).

    That warning fix was itself revised once more the same day: anchoring
    directly to the 2nd-highest score's *value* repeated the same mistake
    BLOCKING_MARGIN's docstring describes — it put the boundary exactly on
    top of one real historical point again. warning now fires past the
    *midpoint* of the data's own largest gap (excluding the top point,
    which blocking owns) — ~4.16, inside the real ~0.68 gap between the
    3rd- and 2nd-highest scores, not sitting on either endpoint. This
    test's pinned numbers were updated to match each revision — see
    checks.py's own comments for the full reasoning, this docstring is the
    abridged version.

    Pinning today's actual derived numbers here means a retrain that changes
    them fails this test loudly, forcing a deliberate look at the new
    values before they ship — rather than either silently drifting forever
    (pure formula, no test) or silently going stale (a hardcoded number with
    only a documentation comment asking someone to remember to update it).
    When this test fails after a real retrain: that is the intended
    checkpoint firing, not a bug -- update the pinned numbers here (and in
    DELIVERY_REPORT.md's "known limitations" section) to the new formula
    output after confirming the new values make sense, don't just bump them
    to whatever silences the assertion."""
    training_z = tec_bridge.load_training_latent()
    n = len(training_z)
    loo_scores = []
    for index in range(n):
        rest = np.delete(training_z, index, axis=0)
        rest_mean = rest.mean(axis=0)
        rest_std = rest.std(axis=0, ddof=1)
        rest_std = np.where(rest_std > 1e-8, rest_std, 1.0)
        score = np.sqrt(np.sum(((training_z[index] - rest_mean) / rest_std) ** 2))
        loo_scores.append(float(score))
    loo_scores.sort()

    assert n == 31, "sample count changed -- re-derive and update the pinned values below"
    training_max = loo_scores[-1]
    assert training_max == pytest.approx(13.32, abs=0.01)
    assert training_max * BLOCKING_MARGIN == pytest.approx(13.32, abs=0.01)
    # 2nd-highest historical score -- kept as a landmark even though it's no
    # longer the warning cutoff itself (see docstring above).
    assert loo_scores[-2] == pytest.approx(4.49, abs=0.01)
    # warning cutoff: midpoint of the largest gap among all-but-the-top
    # score -- inside the real gap, not sitting on either endpoint.
    sub_scores = loo_scores[:-1]
    gaps = [sub_scores[i + 1] - sub_scores[i] for i in range(len(sub_scores) - 1)]
    gap_index = int(np.argmax(gaps))
    warning_threshold = (sub_scores[gap_index] + sub_scores[gap_index + 1]) / 2
    assert warning_threshold == pytest.approx(4.16, abs=0.02)
