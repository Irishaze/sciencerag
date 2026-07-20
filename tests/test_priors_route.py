"""Smoke test for the /sciencerag/priors route (M1-6 stub response)."""

import json
from pathlib import Path

import jsonschema
from fastapi.testclient import TestClient

from sciencerag.app import app

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "priors.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["PriorsResponse"]

client = TestClient(app)


def test_priors_route_returns_schema_valid_response():
    resp = client.post(
        "/sciencerag/priors",
        json={"query": "Bi2Te3 leg length vs COP"},
    )
    assert resp.status_code == 200

    body = resp.json()
    jsonschema.validate(instance=body, schema=RESPONSE_SCHEMA)
    assert body["status"] == "ok"
    assert body["trace_id"].startswith("tr_")


def test_priors_route_rejects_missing_query():
    resp = client.post("/sciencerag/priors", json={})
    assert resp.status_code == 422
