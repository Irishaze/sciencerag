"""Smoke tests for the /sciencerag/ask route (spec §6, M5).

Mocks the LLM call (litellm.completion) and the fallback document-retrieval
call (sciencerag.priors.retrieval.run_query) so this stays fast/free —
same convention as test_priors_route.py. The graph store itself is real
(points GRAPH_PATH at a tmp file), since that part has no external cost.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.ask import pipeline as ask_pipeline
from sciencerag.priors import kg

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sciencerag" / "schemas" / "ask.schema.json"
RESPONSE_SCHEMA = json.loads(SCHEMA_PATH.read_text())["AskResponse"]

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")


def _fake_llm_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_graph_hit_answers_without_fallback(monkeypatch):
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    monkeypatch.setattr(
        ask_pipeline.litellm, "completion", lambda **kwargs: _fake_llm_response("delta_T_max_K is 71.7K.")
    )

    response = client.post(
        "/sciencerag/ask", json={"question": "Bi2Te3 achieves_delta_T_max_K", "max_hits": 5}
    )
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is False
    assert payload["coverage_note"] is None
    assert payload["answer"] == "delta_T_max_K is 71.7K."
    assert len(payload["subgraph"]["nodes"]) == 2
    assert len(payload["subgraph"]["edges"]) == 1
    assert payload["sources"] == [{"type": "kg_triple", "triple_id": payload["subgraph"]["edges"][0]["triple_id"]}]


def test_empty_graph_falls_back_to_document_retrieval(monkeypatch):
    fake_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="p.1"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(answer="From the literature: ...", contexts=[fake_context])
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is True
    assert payload["coverage_note"] is not None
    assert payload["answer"] == "From the literature: ..."
    assert payload["sources"] == [{"type": "paper", "doi": "10.1/fake", "span": "p.1"}]
    assert payload["subgraph"] == {"nodes": [], "edges": []}


def test_missing_question_is_rejected():
    response = client.post("/sciencerag/ask", json={})
    assert response.status_code == 422
