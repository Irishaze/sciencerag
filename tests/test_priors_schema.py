"""Schema validation tests for sciencerag.priors (2 valid + 2 invalid fixtures, spec §3.3)."""

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "priors.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text())
RESPONSE_SCHEMA = SCHEMA["PriorsResponse"]

# -- valid fixtures ------------------------------------------------------

VALID_FULL = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_2026_0713_001",
            "kind": "parameter_range",
            "field": "leg_length_um",
            "value": {"min": 20, "max": 200, "typical": 60},
            "confidence": 0.82,
            "sources": [
                {"type": "paper", "doi": "10.1234/example", "span": "p.4, Fig.3"},
                {"type": "kg_triple", "triple_id": "kg_88213"},
            ],
            "notes": "腿长受制造工艺约束",
        }
    ],
    "coverage": {"internal_hits": 7, "external_hits": 2, "gaps": ["接触电阻数据稀缺"]},
    "trace_id": "tr_test123",
}

VALID_EMPTY_PRIORS = {
    "status": "ok",
    "priors": [],
    "coverage": {"internal_hits": 0, "external_hits": 0},
    "trace_id": "tr_empty",
}

# -- invalid fixtures -----------------------------------------------------

INVALID_MISSING_TRACE_ID = {
    "status": "ok",
    "priors": [],
    "coverage": {"internal_hits": 0, "external_hits": 0},
}

INVALID_BAD_KIND_ENUM = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_bad",
            "kind": "not_a_real_kind",
            "field": "x",
            "value": {},
            "confidence": 0.5,
            "sources": [{"type": "paper", "doi": "10.1/x"}],
        }
    ],
    "coverage": {"internal_hits": 1, "external_hits": 0},
    "trace_id": "tr_bad",
}


@pytest.mark.parametrize("payload", [VALID_FULL, VALID_EMPTY_PRIORS])
def test_valid_priors_response(payload):
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


@pytest.mark.parametrize("payload", [INVALID_MISSING_TRACE_ID, INVALID_BAD_KIND_ENUM])
def test_invalid_priors_response(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
