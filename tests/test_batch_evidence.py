"""Tests for sciencerag.priors.batch_evidence (spec §3.4, M6). Mocks
run_query and litellm.completion — no real PaperQA2/LLM calls, same
convention as the rest of the mocked test suite.
"""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors import batch_evidence
from sciencerag.priors.batch_evidence import CandidateSpec, get_candidate_evidence

client = TestClient(app)


def _fake_context(text: str, doi: str, score: float = 8.0):
    return SimpleNamespace(
        context=text, score=score, text=SimpleNamespace(doc=SimpleNamespace(doi=doi), name="p.1")
    )


def _fake_query_response(contexts):
    return SimpleNamespace(session=SimpleNamespace(contexts=contexts))


def _fake_llm_response(payload):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def test_no_contexts_is_insufficient_coverage(monkeypatch):
    monkeypatch.setattr(batch_evidence, "run_query", lambda q: _fake_query_response([]))
    result = get_candidate_evidence(CandidateSpec(candidate_id="c1", description="short legs maximize COP"))
    assert result.coverage == "insufficient"
    assert result.supporting == result.refuting == result.neutral == []


def test_classifies_supports_and_refutes(monkeypatch):
    contexts = [_fake_context("short legs raise COP", "10.1/s"), _fake_context("long legs raise COP", "10.1/r")]
    monkeypatch.setattr(batch_evidence, "run_query", lambda q: _fake_query_response(contexts))
    monkeypatch.setattr(
        batch_evidence.litellm,
        "completion",
        lambda **kwargs: _fake_llm_response(
            [{"label": "E1", "stance": "supports"}, {"label": "E2", "stance": "refutes"}]
        ),
    )
    result = get_candidate_evidence(CandidateSpec(candidate_id="c1", description="short legs maximize COP"))
    assert result.coverage == "ok"
    assert len(result.supporting) == 1
    assert len(result.refuting) == 1
    assert result.supporting[0].doi == "10.1/s"
    assert result.refuting[0].doi == "10.1/r"


def test_malformed_llm_output_degrades_to_neutral(monkeypatch):
    contexts = [_fake_context("short legs raise COP", "10.1/s")]
    monkeypatch.setattr(batch_evidence, "run_query", lambda q: _fake_query_response(contexts))
    monkeypatch.setattr(
        batch_evidence.litellm,
        "completion",
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        ),
    )
    result = get_candidate_evidence(CandidateSpec(candidate_id="c1", description="x"))
    assert result.coverage == "ok"
    assert len(result.neutral) == 1
    assert result.supporting == result.refuting == []


def test_batch_evidence_route(monkeypatch):
    contexts = [_fake_context("short legs raise COP", "10.1/s")]
    monkeypatch.setattr(batch_evidence, "run_query", lambda q: _fake_query_response(contexts))
    monkeypatch.setattr(
        batch_evidence.litellm,
        "completion",
        lambda **kwargs: _fake_llm_response([{"label": "E1", "stance": "supports"}]),
    )
    response = client.post(
        "/sciencerag/priors/batch_evidence",
        json={"candidates": [{"candidate_id": "c1", "description": "short legs maximize COP"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["candidate_id"] == "c1"
    assert body["results"][0]["coverage"] == "ok"
    assert len(body["results"][0]["supporting"]) == 1


def test_batch_evidence_requires_at_least_one_candidate():
    response = client.post("/sciencerag/priors/batch_evidence", json={"candidates": []})
    assert response.status_code == 422


def test_batch_evidence_rejects_more_than_the_cap(monkeypatch):
    # Regression for a real finding (2026-08-15 adversarial review):
    # get_batch_evidence runs one real PaperQA2 query per candidate,
    # sequentially, with no concurrency cap — an unbounded candidates list
    # let a single request fan out into an unbounded number of billed API
    # calls and tie up the request indefinitely. run_query is monkeypatched
    # to fail loudly if it's ever reached — this must be rejected by
    # request validation before any real work starts.
    from sciencerag.priors import batch_evidence as batch_evidence_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_query must not be reached once the candidate cap is exceeded")

    monkeypatch.setattr(batch_evidence_module, "run_query", _fail_if_called)

    too_many = [
        {"candidate_id": f"c{i}", "description": "x"}
        for i in range(batch_evidence_module.MAX_BATCH_EVIDENCE_CANDIDATES + 1)
    ]
    response = client.post("/sciencerag/priors/batch_evidence", json={"candidates": too_many})
    assert response.status_code == 422
