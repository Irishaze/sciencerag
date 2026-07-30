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


def _fake_priors_response(query: str, allow_external: bool = False) -> tuple[PriorsResponse, int]:
    return (
        PriorsResponse(
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
        ),
        0,
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


def test_priors_route_threads_allow_external_to_retrieval(monkeypatch):
    """M1-15: allow_external must actually reach retrieval.build_priors_response
    (not get silently dropped by the router) so it can be turned into the
    request-acknowledgment gap note — see test_allow_external.py."""
    received = {}

    def _spy(query: str, allow_external: bool = False) -> tuple[PriorsResponse, int]:
        received["allow_external"] = allow_external
        return _fake_priors_response(query, allow_external)

    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _spy)

    resp = client.post(
        "/sciencerag/priors",
        json={"query": "x", "allow_external": True},
    )
    assert resp.status_code == 200
    assert received["allow_external"] is True


def test_priors_route_returns_structured_error_on_unexpected_exception(monkeypatch):
    """If retrieval blows up (network error, litellm timeout, whatever),
    the route must still return our own typed ErrorResponse JSON — never a
    bare FastAPI default 500 (spec §8: always typed, schema-valid errors)."""

    def _boom(query: str, allow_external: bool = False) -> PriorsResponse:
        raise RuntimeError("simulated PaperQA2/LLM failure")

    monkeypatch.setattr(priors_router.retrieval, "build_priors_response", _boom)

    resp = client.post("/sciencerag/priors", json={"query": "will fail"})
    assert resp.status_code == 502

    body = resp.json()
    error_schema = json.loads(SCHEMA_PATH.read_text())["ErrorResponse"]
    jsonschema.validate(instance=body, schema=error_schema)
    assert body["status"] == "error"
    assert body["error"]["category"] == "retrieval_timeout"
    assert "simulated PaperQA2/LLM failure" in body["error"]["message"]
    assert body["trace_id"].startswith("tr_")
