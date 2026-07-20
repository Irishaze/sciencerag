"""Smoke test for the /sciencerag/priors route.

Mocks out real PaperQA2 retrieval (sciencerag.priors.router.retrieval) so
this stays fast/free — it's testing routing + schema plumbing, not
retrieval quality. Real end-to-end retrieval is checked separately by
scripts/verify_paperqa.py and the M1-17/18 regression fixtures.
"""

import json
from pathlib import Path

import jsonschema
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors import router as priors_router
from sciencerag.priors.models import Coverage, Prior, PriorsResponse, SourcePaper

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "priors.schema.json"
)
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["PriorsResponse"]

client = TestClient(app)


def _fake_priors_response(query: str) -> PriorsResponse:
    return PriorsResponse(
        priors=[
            Prior(
                prior_id="pr_fake_0001",
                kind="parameter_range",
                field="general_finding",
                value={"summary": f"fake evidence for: {query}"},
                confidence=0.8,
                sources=[SourcePaper(doi="10.0000/fake", span="pages 1-2")],
            )
        ],
        coverage=Coverage(internal_hits=1, external_hits=0, gaps=[]),
        trace_id="tr_fake_test",
    )


def test_priors_route_returns_schema_valid_response(monkeypatch):
    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _fake_priors_response)

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
