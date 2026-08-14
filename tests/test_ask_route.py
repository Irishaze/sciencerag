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
    # Regression: SubgraphNode used to only declare id/kind, so
    # subgraph_from_triples()'s label/entity_type were silently dropped by
    # response_model validation before ever reaching the frontend —
    # GraphView.tsx renders node.label and colors by node.entity_type, so a
    # dropped field meant every node showed its opaque id instead of a
    # human-readable name.
    entity_node = next(n for n in payload["subgraph"]["nodes"] if n["kind"] == "entity")
    assert entity_node["label"] == "Bi2Te3 single-stage TEC"
    assert entity_node["entity_type"]


def test_evidence_detail_is_included_in_the_prompt(monkeypatch):
    # Regression for the 2026-08-14/15 "方案B" wiring: two triples with the
    # same confidence can have very different real support (a 0.8% match
    # vs a 4.9% one) — the LLM must actually see that distinction, not just
    # a bare confidence number.
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.85,
        run_id="run_1",
        sources=[],
        evidence_detail={"verdict": "consistent", "relative_deviation": 0.008, "benchmark_case_id": "sample_02.docx"},
    )
    captured_prompts = []

    def _fake_completion(**kwargs):
        captured_prompts.append(kwargs["messages"][1]["content"])
        return _fake_llm_response("delta_T_max_K is 71.7K.")

    monkeypatch.setattr(ask_pipeline.litellm, "completion", _fake_completion)

    response = client.post("/sciencerag/ask", json={"question": "Bi2Te3 achieves_delta_T_max_K"})
    assert response.status_code == 200
    assert "evidence_detail" in captured_prompts[0]
    assert "0.8%" in captured_prompts[0]
    assert "sample_02.docx" in captured_prompts[0]


def test_multi_design_graph_answer_reports_truncation_honestly(monkeypatch):
    # Regression for the bug found 2026-08-11: a generic question against a
    # multi-design graph used to silently truncate at max_hits with no
    # signal, and the LLM confidently claimed "no other data exists" for
    # designs that were simply never retrieved into its context. With
    # multiple designs and max_hits below the total, the coverage note (and
    # the prompt handed to the LLM) must say so explicitly — and each
    # included design's triples must arrive complete, not partially cut.
    for i in range(3):
        kg.add_triple(
            subject="Bi2Te3 single-stage TEC",
            relation="achieves_delta_T_max_K",
            object_value=70.0 + i,
            object_unit="K",
            conditions={"leg_length": 0.5 + i * 0.1},
            confidence=0.7,
            run_id="run_1",
            sources=[],
        )
        kg.add_triple(
            subject="Bi2Te3 single-stage TEC",
            relation="achieves_optimal_current_A",
            object_value=5.0 + i,
            object_unit="A",
            conditions={"leg_length": 0.5 + i * 0.1},
            confidence=0.7,
            run_id="run_1",
            sources=[],
        )

    captured_prompts = []

    def _fake_completion(**kwargs):
        captured_prompts.append(kwargs["messages"][1]["content"])
        return _fake_llm_response("some answer")

    monkeypatch.setattr(ask_pipeline.litellm, "completion", _fake_completion)

    response = client.post(
        "/sciencerag/ask",
        json={"question": "Bi2Te3 single-stage TEC achieves_delta_T_max_K", "max_hits": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is False
    assert payload["coverage_note"] is not None
    assert "3 matching design" in payload["coverage_note"]
    assert "2" in payload["coverage_note"]

    # each included design's triples arrive together, never a partial cut
    edges = payload["subgraph"]["edges"]
    assert len(edges) == 4  # 2 designs x 2 relations each, not 2 rows total

    assert "only 2 of 3 matching design" in captured_prompts[0]


def test_ranking_question_answers_from_graph_with_ranking_block(monkeypatch):
    # Regression for the 2026-08-11 first-principles finding: a query with
    # a superlative word ("最高") used to be treated as an ordinary keyword,
    # so every candidate design scored roughly equal relevance and no
    # actual comparison ever ran — the LLM was handed an unordered dump and
    # left to guess which one was "best". A real ranking (sorted by
    # object_value, not text overlap) must now be computed and injected
    # into the prompt as its own explicit block.
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=58.2,
        object_unit="K",
        conditions={"leg_length": 0.5},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最大温差",
    )
    highest, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=81.5,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最大温差",
    )

    captured_prompts = []

    def _fake_completion(**kwargs):
        captured_prompts.append(kwargs["messages"][1]["content"])
        return _fake_llm_response("81.5K")

    monkeypatch.setattr(ask_pipeline.litellm, "completion", _fake_completion)

    response = client.post("/sciencerag/ask", json={"question": "哪个设计的最大温差最高"})
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is False

    prompt = captured_prompts[0]
    assert "Ranking:" in prompt
    assert f"[{highest.triple_id}]" in prompt
    # the ranked #1 triple must actually be handed to the LLM as a source/
    # subgraph edge, not just mentioned in prose
    assert any(s["triple_id"] == highest.triple_id for s in payload["sources"])


def test_empty_graph_falls_back_to_document_retrieval(monkeypatch):
    fake_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="p.1"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(answer="From the literature: p.1 says so.", contexts=[fake_context])
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is True
    assert payload["coverage_note"] is not None
    assert payload["answer"] == "From the literature: p.1 says so."
    assert payload["sources"] == [{"type": "paper", "doi": "10.1/fake", "span": "p.1"}]
    assert payload["subgraph"] == {"nodes": [], "edges": []}


def test_fallback_sources_exclude_retrieved_but_uncited_contexts(monkeypatch):
    # Regression for a real mismatch found 2026-08-10: gather_evidence
    # routinely retrieves more context than generate_answer ends up citing,
    # and the old code listed every retrieved context as a "source" — a
    # live response had 2 of 3 listed sources absent from the answer text
    # entirely. Only contexts whose docname actually appears in the final
    # answer should be listed.
    cited_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/cited"), name="cited2024paper pages 1-2"),
        score=9.0,
    )
    uncited_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/uncited"), name="uncited2025paper pages 3-4"),
        score=7.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(
            answer="The finding comes from cited2024paper pages 1-2.",
            contexts=[cited_context, uncited_context],
        )
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    payload = response.json()
    assert payload["sources"] == [{"type": "paper", "doi": "10.1/cited", "span": "cited2024paper pages 1-2"}]


def test_fallback_backfills_bare_citation_key_paperqa_left_unresolved(monkeypatch):
    # Regression for the bug found 2026-08-10: paper-qa's own
    # populate_formatted_answers_and_bib_from_raw_answer only resolves a
    # pqac-xxxxxxxx key into its source docname when the key sits inside a
    # parenthetical exactly matching its CITATION_KEY_CONSTRAINTS grammar.
    # Our custom answer_length formatting instructions (priors/retrieval.py)
    # make the model more likely to drop that exact format, so a raw
    # pqac-xxxxxxxx key can leak straight into session.answer. We backfill
    # it ourselves as a second, more lenient pass.
    fake_context = SimpleNamespace(
        id="pqac-5ca94a10",
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="shan2024designandoptimization"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(
            answer="The optimal current is I_opt = alpha*Tc/R pqac-5ca94a10.",
            contexts=[fake_context],
        )
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert "pqac-" not in payload["answer"]
    assert "shan2024designandoptimization" in payload["answer"]


def test_fallback_drops_hallucinated_citation_key(monkeypatch):
    # A pqac key with no matching context (the model invented it, or it
    # refers to a context that didn't make the final cut) must be dropped
    # cleanly rather than shown to the user as an opaque hash.
    fake_context = SimpleNamespace(
        id="pqac-5ca94a10",
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="shan2024designandoptimization"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(
            answer="Some claim (pqac-00000000) with no matching context.",
            contexts=[fake_context],
        )
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    payload = response.json()
    assert "pqac-" not in payload["answer"]
    assert payload["answer"] == "Some claim  with no matching context."


def test_fallback_answer_gets_unicode_cleanup(monkeypatch):
    fake_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="p.1"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        # U+2011 NON-BREAKING HYPHEN, the character reported as rendering
        # as a tofu box in "发电‑制冷".
        session=SimpleNamespace(answer="发电‑制冷 findings.", contexts=[fake_context])
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post("/sciencerag/ask", json={"question": "some question with no graph coverage"})
    payload = response.json()
    assert payload["answer"] == "发电-制冷 findings."


def test_weak_keyword_overlap_falls_back_instead_of_answering_from_graph(monkeypatch):
    # Regression for the bug found 2026-08-10: a query sharing only one
    # incidental token (e.g. "TEC") with a triple's subject used to score
    # nonzero relevance against *every* triple for that subject and get
    # treated as a real graph hit, even though it has nothing to do with the
    # question actually asked. Six triples about one demo geometry, none of
    # which mention "input current optimization", must not out-rank a
    # fallback to the literature corpus just because they all share "TEC".
    for relation, value, unit in [
        ("achieves_delta_T_max_K", 71.741, "K"),
        ("achieves_optimal_current_A", 6.4831, "A"),
        ("achieves_optimal_voltage_V", 0.27485, "V"),
        ("achieves_total_resistance_ohm", 0.042394, "ohm"),
        ("achieves_max_heat_dissipation_W", 1.0862, "W"),
        ("achieves_figure_of_merit_1_per_K", 0.0023844, "1/K"),
    ]:
        kg.add_triple(
            subject="Bi2Te3 single-stage TEC",
            relation=relation,
            object_value=value,
            object_unit=unit,
            conditions={"leg_length": 0.7984},
            confidence=0.7,
            run_id="demo_run",
            sources=[],
        )
    fake_context = SimpleNamespace(
        text=SimpleNamespace(doc=SimpleNamespace(doi="10.1/fake"), name="p.1"),
        score=8.0,
    )
    fake_response = SimpleNamespace(
        session=SimpleNamespace(answer="From the literature: I_opt = alpha*Tc/R", contexts=[fake_context])
    )
    monkeypatch.setattr(ask_pipeline, "run_query", lambda query: fake_response)

    response = client.post(
        "/sciencerag/ask",
        json={"question": "How should input current in a TEC be optimized to maximize cooling capacity?"},
    )
    assert response.status_code == 200
    payload = response.json()
    jsonschema.validate(instance=payload, schema=RESPONSE_SCHEMA)
    assert payload["fallback_used"] is True
    assert "minimum relevance" in payload["coverage_note"]
    assert payload["answer"] == "From the literature: I_opt = alpha*Tc/R"


def test_missing_question_is_rejected():
    response = client.post("/sciencerag/ask", json={})
    assert response.status_code == 422


def test_graph_endpoint_empty_graph():
    response = client.get("/sciencerag/graph")
    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": []}


def test_graph_endpoint_returns_every_triple_not_just_one_question():
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=0.5,
        object_unit="A",
        conditions={},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    response = client.get("/sciencerag/graph")
    body = response.json()
    assert len(body["edges"]) == 2
    assert {edge["relation"] for edge in body["edges"]} == {
        "achieves_delta_T_max_K",
        "achieves_optimal_current_A",
    }
