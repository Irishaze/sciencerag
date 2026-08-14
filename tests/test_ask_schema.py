"""Schema fixture tests for sciencerag.ask (spec §6)."""

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "ask.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
REQUEST_SCHEMA = SCHEMA["AskRequest"]
RESPONSE_SCHEMA = SCHEMA["AskResponse"]

VALID_RESPONSE = {
    "status": "ok",
    "answer": "delta_T_max_K is 71.7K.",
    "subgraph": {
        "nodes": [
            {
                "id": "tec_abc123",
                "kind": "entity",
                "label": "Bi2Te3 single-stage TEC",
                "entity_type": "TECDesign",
            }
        ],
        "edges": [],
    },
    "sources": [{"type": "kg_triple", "triple_id": "kg_1"}],
    "fallback_used": False,
    "coverage_note": None,
    "trace_id": "tr_1",
}

INVALID_BAD_NODE_KIND = {
    **VALID_RESPONSE,
    "subgraph": {"nodes": [{"id": "x", "kind": "not_a_real_kind"}], "edges": []},
}

INVALID_MISSING_ANSWER = {k: v for k, v in VALID_RESPONSE.items() if k != "answer"}


def test_valid_ask_response():
    jsonschema.validate(instance=VALID_RESPONSE, schema=RESPONSE_SCHEMA)


@pytest.mark.parametrize("payload", [INVALID_BAD_NODE_KIND, INVALID_MISSING_ANSWER])
def test_invalid_ask_response(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


def test_valid_ask_request():
    jsonschema.validate(instance={"question": "what leg length maximizes COP?"}, schema=REQUEST_SCHEMA)


def test_ask_request_missing_question_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={}, schema=REQUEST_SCHEMA)


@pytest.mark.parametrize("bad_max_hits", [0, -1, 51])
def test_ask_request_max_hits_out_of_range_is_invalid(bad_max_hits):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"question": "x", "max_hits": bad_max_hits}, schema=REQUEST_SCHEMA)
