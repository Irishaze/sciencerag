"""Schema/model tests for sciencerag.validate (spec §4.5).

jsonschema fixtures for the wire format, plus direct Pydantic tests for the
blocked=>empty invariant that jsonschema's structural validation can't
express (it's a cross-field constraint enforced by UpdatePackage's
model_validator, not the JSON Schema itself).
"""

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from sciencerag.validate.models import UpdatePackage, ValidateResponse

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "validate.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text())
REQUEST_SCHEMA = SCHEMA["ValidateRequest"]
RESPONSE_SCHEMA = SCHEMA["ValidateResponse"]

VALID_RESPONSE = {
    "status": "ok",
    "anomalies": [
        {"check": "energy_balance", "severity": "info", "evidence": {}},
        {"check": "pde_residual", "severity": "warning", "evidence": {"ratio_to_baseline_max": 3.1}},
        {"check": "ood", "severity": "info", "evidence": {}},
    ],
    "evaluation": {"verdict": "consistent", "deviations": [], "sources": []},
    "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": False},
    "trace_id": "tr_test",
}

INVALID_BAD_CHECK_ENUM = {
    **VALID_RESPONSE,
    "anomalies": [{"check": "not_a_real_check", "severity": "info", "evidence": {}}],
}

INVALID_BAD_SEVERITY_ENUM = {
    **VALID_RESPONSE,
    "anomalies": [{"check": "ood", "severity": "catastrophic", "evidence": {}}],
}

INVALID_MISSING_TRACE_ID = {k: v for k, v in VALID_RESPONSE.items() if k != "trace_id"}


@pytest.mark.parametrize("payload", [VALID_RESPONSE])
def test_valid_validate_response(payload):
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


@pytest.mark.parametrize(
    "payload", [INVALID_BAD_CHECK_ENUM, INVALID_BAD_SEVERITY_ENUM, INVALID_MISSING_TRACE_ID]
)
def test_invalid_validate_response(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


def test_valid_validate_request():
    jsonschema.validate(instance={"run_id": "run_1"}, schema=REQUEST_SCHEMA)


def test_validate_request_missing_run_id_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={}, schema=REQUEST_SCHEMA)


@pytest.mark.parametrize("bad_n_pairs", [0, 21, -1])
def test_validate_request_n_pairs_out_of_range_is_invalid(bad_n_pairs):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"run_id": "run_1", "n_pairs": bad_n_pairs}, schema=REQUEST_SCHEMA
        )


# -- blocked=>empty invariant (model-level, not expressible in plain jsonschema) --


def test_blocked_true_requires_empty_update_package():
    UpdatePackage(surrogate_update=None, kg_candidates=[], blocked=True)  # ok


def test_blocked_true_with_surrogate_update_is_rejected():
    with pytest.raises(ValidationError):
        UpdatePackage(surrogate_update={"note": "should not be allowed"}, kg_candidates=[], blocked=True)


def test_blocked_true_with_kg_candidates_is_rejected():
    with pytest.raises(ValidationError):
        UpdatePackage(surrogate_update=None, kg_candidates=[{"triple": "x"}], blocked=True)


def test_blocked_false_allows_populated_update_package():
    UpdatePackage(
        surrogate_update={"hyperparameter_direction": "ok"},
        kg_candidates=[
            {
                "subject": "Bi2Te3 single-stage TEC",
                "relation": "achieves_delta_T_max_K",
                "object_value": 50.0,
                "confidence": 0.7,
                "run_id": "run_1",
                "dedup_status": "new",
            }
        ],
        blocked=False,
    )
