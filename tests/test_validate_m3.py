"""Unit tests for M3 (spec §4.3/§4.4): sciencerag/validate/finetune.py and
sciencerag/validate/kg_candidates.py.

kg.query_kg is still a stub that always returns zero hits (sciencerag/
priors/kg.py's documented cold-start state) — the router-level tests in
test_validate_route.py can therefore only ever exercise dedup_status="new".
This file monkeypatches query_kg to also exercise the
"duplicate_confirmed" branch, so that logic isn't dead code today and
doesn't silently break later when kg.py gets a real graph behind it.
"""

from sciencerag.priors.kg import KGHit
from sciencerag.validate import kg_candidates as kg_candidates_module
from sciencerag.validate.finetune import suggest_surrogate_update
from sciencerag.validate.kg_candidates import extract_kg_candidates
from sciencerag.validate.models import Anomaly, Evaluation, ValidateRequest


def _request(**overrides) -> ValidateRequest:
    defaults = {"run_id": "run_unit", "design_parameters": {}, "n_pairs": 1, "priors": []}
    return ValidateRequest.model_validate({**defaults, **overrides})


# -- finetune.py ----------------------------------------------------------


def test_no_signal_returns_none():
    request = _request()
    anomalies = [Anomaly(check="ood", severity="info", evidence={})]
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    assert suggest_surrogate_update(request, anomalies, evaluation) is None


def test_warning_anomaly_produces_loss_reweighting():
    request = _request()
    anomalies = [Anomaly(check="energy_balance", severity="warning", evidence={"ratio": 3.0})]
    evaluation = Evaluation(verdict="consistent", deviations=[], sources=[])
    suggestion = suggest_surrogate_update(request, anomalies, evaluation)
    assert suggestion is not None
    assert suggestion.loss_reweighting == {"energy_balance": 1.5}
    assert "interface_weight" in suggestion.hyperparameter_direction


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
    assert suggestion.loss_reweighting == {}
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
    assert len(candidates) == 1
    assert candidates[0].confidence == 0.4
    assert candidates[0].object_unit == "K"


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
