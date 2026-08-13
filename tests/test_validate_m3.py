"""Unit tests for M3 (spec §4.3/§4.4): sciencerag/validate/finetune.py and
sciencerag/validate/kg_candidates.py.

kg.py has a real JSON-backed graph since M5 (sciencerag/priors/kg.py) —
GRAPH_PATH is redirected at a tmp file for every test in this file so
dedup_status stays deterministic regardless of what's actually been
written to the real data/kg/graph.json (e.g. by scripts/demo_end_to_end.py
or a real approval run). The "duplicate_confirmed"-specific tests still
monkeypatch query_kg directly for precise control over what a "hit" looks
like, on top of that isolation.
"""

import pytest

from sciencerag.priors import kg
from sciencerag.priors.kg import KGHit
from sciencerag.validate import kg_candidates as kg_candidates_module
from sciencerag.validate.finetune import suggest_surrogate_update
from sciencerag.validate.kg_candidates import extract_kg_candidates
from sciencerag.validate.models import Anomaly, Evaluation, ValidateRequest


@pytest.fixture(autouse=True)
def _tmp_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")


@pytest.fixture(autouse=True)
def _no_ontology_by_default(monkeypatch):
    # Regression for a real incident found 2026-08-11: once a real
    # data/kg/ontology.json existed on disk (scripts/generate_kg_ontology.py
    # had actually been run), these tests silently stopped being free/fast
    # — extract_kg_candidates() started making real classify_relation/
    # describe_relation LLM calls on every run, because load_ontology()
    # reads that file directly with no test-side mocking. Tests must not
    # depend on incidental host filesystem state; default every test in
    # this file to the cold-start "no ontology yet" path, and let the
    # specific tests below that want to exercise the ontology-present path
    # override this with their own explicit (still-mocked) monkeypatch.
    monkeypatch.setattr(kg_candidates_module, "load_ontology", lambda: None)


def _request(**overrides) -> ValidateRequest:
    defaults = {"run_id": "run_unit", "design_parameters": {}, "n_pairs": 1, "priors": []}
    return ValidateRequest.model_validate({**defaults, **overrides})


# -- finetune.py ----------------------------------------------------------


def test_no_signal_returns_none():
    request = _request()
    anomalies = [Anomaly(check="ood", severity="info", evidence={})]
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    assert suggest_surrogate_update(request, anomalies, evaluation) is None


def test_warning_anomaly_produces_hyperparameter_direction():
    request = _request()
    anomalies = [Anomaly(check="ood", severity="warning", evidence={"mahalanobis_distance": 3.0})]
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    suggestion = suggest_surrogate_update(request, anomalies, evaluation)
    assert suggestion is not None
    assert "Sobol" in suggestion.hyperparameter_direction


def test_deviation_without_warning_anomaly_has_no_hyperparameter_direction():
    """Error-driven-only signal (no physics-check warning) shouldn't invent
    a hyperparameter direction it has no basis for."""
    from sciencerag.validate.models import Deviation

    request = _request()
    anomalies = [Anomaly(check="ood", severity="info", evidence={})]
    evaluation = Evaluation(
        verdict="deviation_found",
        deviations=[
            Deviation(
                field="leg_length",
                source="prior_comparison",
                reference_id="pr_1",
                actual=0.1,
                reference_min=1.0,
                reference_max=2.0,
                verdict="deviation",
            )
        ],
        sources=[],
    )
    suggestion = suggest_surrogate_update(request, anomalies, evaluation)
    assert suggestion is not None
    assert "no specific hyperparameter direction" in suggestion.hyperparameter_direction
    assert suggestion.recommended_training_samples[0].region == "prior_comparison:leg_length"


# -- kg_candidates.py -------------------------------------------------------


def test_deviation_found_extracts_nothing():
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="deviation_found", deviations=[], sources=[])
    assert extract_kg_candidates(request, evaluation) == []


def test_no_scalar_results_extracts_nothing():
    request = _request()
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    assert extract_kg_candidates(request, evaluation) == []


def test_insufficient_benchmark_still_extracts_at_lower_confidence():
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="insufficient_benchmark", deviations=[], sources=[])
    candidates = extract_kg_candidates(request, evaluation)
    # 1 numeric fact (achieves_delta_T_max_K) + 2 structural link candidates
    # (SimulationRun -> TECDesign, SimulationRun -> Material) — see
    # kg_candidates.py's own comment on why those two always get appended.
    assert len(candidates) == 3
    numeric = next(c for c in candidates if c.object_value is not None)
    assert numeric.confidence == 0.4
    assert numeric.object_unit == "K"
    links = [c for c in candidates if c.object_entity_id is not None]
    assert len(links) == 2
    assert {c.relation for c in links} == {"SIMULATION_USES_DESIGN", "SIMULATION_USES_MATERIAL"}


def test_high_relevance_kg_hit_marks_duplicate(monkeypatch):
    monkeypatch.setattr(
        kg_candidates_module,
        "query_kg",
        lambda query: [KGHit(triple_id="kg_1", text=query, relevance=0.95)],
    )
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    candidates = extract_kg_candidates(request, evaluation)
    assert candidates[0].dedup_status == "duplicate_confirmed"


def test_low_relevance_kg_hit_stays_new(monkeypatch):
    monkeypatch.setattr(
        kg_candidates_module,
        "query_kg",
        lambda query: [KGHit(triple_id="kg_1", text=query, relevance=0.2)],
    )
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    candidates = extract_kg_candidates(request, evaluation)
    assert candidates[0].dedup_status == "new"


def test_no_ontology_falls_back_to_default_entity_type(monkeypatch):
    # No data/kg/ontology.json in this tmp-isolated test env — must not
    # error or trigger a real LLM call, just fall back gracefully (same
    # default add_triple() itself uses when no entity_type is supplied).
    monkeypatch.setattr(kg_candidates_module, "load_ontology", lambda: None)
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    candidates = extract_kg_candidates(request, evaluation)
    assert candidates[0].entity_type == "Unclassified"


def test_ontology_present_uses_classification(monkeypatch):
    monkeypatch.setattr(kg_candidates_module, "load_ontology", lambda: object())
    monkeypatch.setattr(
        kg_candidates_module, "classify_relation", lambda relation, ontology: "TECDesign"
    )
    monkeypatch.setattr(
        kg_candidates_module, "describe_relation", lambda relation, ontology: "最大温差"
    )
    request = _request(scalar_results={"delta_T_max_K": 50.0})
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    candidates = extract_kg_candidates(request, evaluation)
    assert candidates[0].entity_type == "TECDesign"
    assert candidates[0].relation_description == "最大温差"
