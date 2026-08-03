"""Schema fixture tests for sciencerag.report (spec §5)."""

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "report.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text())
REQUEST_SCHEMA = SCHEMA["ReportRequest"]
RESPONSE_SCHEMA = SCHEMA["ReportResponse"]

VALID_RESPONSE = {
    "status": "ok",
    "run_id": "run_1",
    "generated_at": "2026-08-03T00:00:00+00:00",
    "objective_and_constraints": {"objective": "maximize_cop", "constraints": {}},
    "spec_summary": {"leg_length": 1.0},
    "key_results": [{"field": "delta_T_max_K", "value": 70.0, "unit": "K", "confidence_label": "high"}],
    "literature_comparison": {"verdict": "consistent", "deviations": [], "sources": []},
    "anomalies_and_cautions": [],
    "update_proposal_summary": {"surrogate_update": None, "kg_candidates": [], "blocked": False},
    "citations": [],
    "markdown": "# Report",
    "trace_id": "tr_1",
}

INVALID_BAD_CONFIDENCE_LABEL = {
    **VALID_RESPONSE,
    "key_results": [{"field": "x", "value": 1.0, "confidence_label": "super_confident"}],
}

INVALID_MISSING_RUN_ID = {k: v for k, v in VALID_RESPONSE.items() if k != "run_id"}


def test_valid_report_response():
    jsonschema.validate(instance=VALID_RESPONSE, schema=RESPONSE_SCHEMA)


@pytest.mark.parametrize("payload", [INVALID_BAD_CONFIDENCE_LABEL, INVALID_MISSING_RUN_ID])
def test_invalid_report_response(payload):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)


def test_valid_report_request_minimal():
    jsonschema.validate(
        instance={
            "run_id": "run_1",
            "evaluation": {"verdict": "consistent", "deviations": [], "sources": []},
            "update_package": {"surrogate_update": None, "kg_candidates": [], "blocked": False},
        },
        schema=REQUEST_SCHEMA,
    )


def test_report_request_missing_evaluation_is_invalid():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"run_id": "run_1"}, schema=REQUEST_SCHEMA)
