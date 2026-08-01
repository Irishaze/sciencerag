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
REQUEST_SCHEMA = SCHEMA["PriorsRequest"]

# -- valid fixtures ------------------------------------------------------

VALID_FULL = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_2026_0713_001",
            "kind": "parameter_range",
            "field": "leg_length_um",
            "value": {"field_name": "leg_length_um", "min": 20, "max": 200, "typical": 60, "unit": "um"},
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

# related_fields (spec §3.6) — a scaling_relationship
# prior with null `field`, expressing a relation between two sim-contract
# parameters instead of a single one.
VALID_RELATED_FIELDS = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_2026_0731_002",
            "kind": "scaling_relationship",
            "field": None,
            "related_fields": ["leg_length", "leg_width"],
            "value": {"x": "leg_length", "y": "leg_width", "direction": "positive"},
            "notes": "optimal leg_length depends on leg_width",
            "confidence": 0.7,
            "sources": [{"type": "paper", "doi": "10.1234/example", "span": "p.5, Eq.12"}],
        }
    ],
    "coverage": {"internal_hits": 3, "external_hits": 0, "gaps": []},
    "trace_id": "tr_related_fields",
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

# Spec principle: "every scientific claim must carry a citation" — a Prior
# with zero sources violates that even if every other field is well-formed.
INVALID_ZERO_SOURCES = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_no_sources",
            "kind": "parameter_range",
            "field": "x",
            "value": {"field_name": "x", "typical": 1, "unit": "mm"},
            "confidence": 0.9,
            "sources": [],
        }
    ],
    "coverage": {"internal_hits": 1, "external_hits": 0},
    "trace_id": "tr_no_sources",
}

INVALID_CONFIDENCE_OUT_OF_RANGE = {
    "status": "ok",
    "priors": [
        {
            "prior_id": "pr_overconfident",
            "kind": "parameter_range",
            "field": "x",
            "value": {"field_name": "x", "typical": 1, "unit": "mm"},
            "confidence": 1.5,
            "sources": [{"type": "paper", "doi": "10.1/x"}],
        }
    ],
    "coverage": {"internal_hits": 1, "external_hits": 0},
    "trace_id": "tr_overconfident",
}


@pytest.mark.parametrize("payload", [VALID_FULL, VALID_RELATED_FIELDS, VALID_EMPTY_PRIORS])
def test_valid_priors_response(payload):
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


@pytest.mark.parametrize(
    "payload",
    [
        INVALID_MISSING_TRACE_ID,
        INVALID_BAD_KIND_ENUM,
        INVALID_ZERO_SOURCES,
        INVALID_CONFIDENCE_OUT_OF_RANGE,
    ],
)
def test_invalid_priors_response(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


# -- PriorsRequest schema (previously untested at the jsonschema level —
# only exercised indirectly via FastAPI/Pydantic in test_priors_route.py) --


def test_valid_priors_request():
    jsonschema.validate(
        instance={"query": "Bi2Te3 leg length vs COP"},
        schema=REQUEST_SCHEMA,
    )


def test_priors_request_missing_query_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={}, schema=REQUEST_SCHEMA)


@pytest.mark.parametrize("bad_max_priors", [0, -5])
def test_priors_request_nonpositive_max_priors_is_invalid(bad_max_priors):
    """max_priors previously had no lower bound — 0 or a negative count
    would pass schema validation despite being a nonsensical request."""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"query": "x", "max_priors": bad_max_priors},
            schema=REQUEST_SCHEMA,
        )


def test_priors_request_max_priors_wrong_type_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"query": "x", "max_priors": "5"},
            schema=REQUEST_SCHEMA,
        )


def test_priors_request_allow_external_wrong_type_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"query": "x", "allow_external": "true"},
            schema=REQUEST_SCHEMA,
        )


def test_priors_request_non_numeric_constraint_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"query": "x", "task_context": {"constraints": {"heat_load_w": "high"}}},
            schema=REQUEST_SCHEMA,
        )
