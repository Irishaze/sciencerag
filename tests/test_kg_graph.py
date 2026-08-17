"""Tests for the real NetworkX/JSON-backed graph store (spec §3.2/§4.4/§6,
M5). Points GRAPH_PATH at a tmp file per test so these don't touch/depend
on the real data/kg/graph.json.
"""

import json
import threading
from types import SimpleNamespace

import pytest

from sciencerag.priors import kg


def _reset_graph_path(tmp_path, monkeypatch):
    path = tmp_path / "graph.json"
    monkeypatch.setattr(kg, "GRAPH_PATH", path)
    return path


def _fake_llm_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _stub_ranking_classification(monkeypatch, is_ranking, relation=None, direction=None):
    """rank_kg_entities now decides ranking intent + field via one LLM call
    (sciencerag.priors.kg._classify_ranking_query) instead of the old
    keyword word-list — these tests stub that call so they stay fast/free
    and exercise this module's own sort/grouping logic, not real model
    behavior."""
    payload = json.dumps({"is_ranking": is_ranking, "direction": direction, "relation": relation})

    def fake_completion(*args, **kwargs):
        return _fake_llm_response(payload)

    monkeypatch.setattr(kg.litellm, "completion", fake_completion)


def test_empty_graph_returns_no_hits(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    assert kg.query_kg("Bi2Te3 delta_T_max_K") == []


def test_add_triple_then_query_finds_it(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    triple, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8, "n_pairs": 1.0},
        confidence=0.7,
        run_id="run_1",
        sources=[kg.KGSource(type="run", run_id="run_1")],
    )
    assert status == "added"
    hits = kg.query_kg("Bi2Te3 achieves_delta_T_max_K")
    assert hits and hits[0].triple_id == triple.triple_id


def test_matching_value_merges_and_appends_run_id(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    first, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.8,  # within 2% tolerance of 71.7
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.6,
        run_id="run_2",
        sources=[],
    )
    assert status == "merged"
    assert second.triple_id == first.triple_id
    assert set(second.run_ids) == {"run_1", "run_2"}
    assert len(kg._load_triples()) == 1


def test_merge_upgrades_confidence_when_new_evidence_is_stronger(tmp_path, monkeypatch):
    # Regression for a real gap: the first run to write a triple can land
    # via kg_candidates.py's flat fallback (no matching benchmark case that
    # day), and every later run that re-confirms the same fact with a real
    # per-field deviation behind it used to be silently discarded — the
    # triple stayed frozen at its first, weaker confidence forever.
    _reset_graph_path(tmp_path, monkeypatch)
    first, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,  # flat-fallback value, no evidence_detail
        run_id="run_1",
        sources=[],
    )
    assert first.evidence_detail is None
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.8,  # within 2% tolerance of 71.7
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.85,  # stronger: real per-field deviation this time
        run_id="run_2",
        sources=[],
        evidence_detail={"verdict": "consistent", "relative_deviation": 0.01, "benchmark_case_id": "bench_1"},
    )
    assert status == "merged"
    assert second.confidence == 0.85
    assert second.evidence_detail == {"verdict": "consistent", "relative_deviation": 0.01, "benchmark_case_id": "bench_1"}


def test_merge_does_not_downgrade_confidence_when_new_evidence_is_weaker(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.85,
        run_id="run_1",
        sources=[],
        evidence_detail={"verdict": "consistent", "relative_deviation": 0.01, "benchmark_case_id": "bench_1"},
    )
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.8,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.5,  # weaker — must not overwrite the stronger evidence already stored
        run_id="run_2",
        sources=[],
    )
    assert status == "merged"
    assert second.confidence == 0.85
    assert second.evidence_detail == {"verdict": "consistent", "relative_deviation": 0.01, "benchmark_case_id": "bench_1"}


def test_conflicting_value_is_flagged_not_overwritten(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    first, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=200.0,  # way outside tolerance
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_2",
        sources=[],
    )
    assert status == "conflict"
    assert second.conflicts_with == first.triple_id
    triples = kg._load_triples()
    assert len(triples) == 2
    assert {t.object_value for t in triples} == {71.7, 200.0}


def test_get_subgraph_returns_nodes_and_edges(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
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
    subgraph = kg.get_subgraph(["Bi2Te3 single-stage TEC"])
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["relation"] == "achieves_delta_T_max_K"


def test_value_node_label_is_not_doubled_with_relation_description(tmp_path, monkeypatch):
    # Regression for the bug found 2026-08-11: the value node's own label
    # used to be prefixed with relation_description too (e.g. "最多能降温
    # 多少度：58.2K"), and the frontend separately prepends the edge's own
    # description when displaying a node's connections or an edge's
    # detail — together that rendered as "最多能降温多少度：最多能降温
    # 多少度：58.2K". The value node's label must be just the number; the
    # description belongs on the edge only.
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=58.2,
        object_unit="K",
        conditions={},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最多能降温多少度",
    )
    subgraph = kg.full_graph()
    value_node = next(n for n in subgraph["nodes"] if n["kind"] == "value")
    assert value_node["label"] == "58.2K"
    assert subgraph["edges"][0]["description"] == "最多能降温多少度"


def test_subgraph_nodes_carry_label_and_do_not_collide_on_shared_values(tmp_path, monkeypatch):
    # Regression for a latent bug found 2026-08-11 while designing the
    # entity_id fix: value-nodes used to be keyed by bare
    # f"{relation}={value}{unit}", so two different entities that happen to
    # produce the identical relation+value would render as one shared node.
    _reset_graph_path(tmp_path, monkeypatch)
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
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,  # same relation+value, different design
        object_unit="K",
        conditions={"leg_length": 0.5},
        confidence=0.7,
        run_id="run_2",
        sources=[],
    )
    subgraph = kg.full_graph()
    entity_nodes = [n for n in subgraph["nodes"] if n["kind"] == "entity"]
    value_nodes = [n for n in subgraph["nodes"] if n["kind"] == "value"]
    assert len(entity_nodes) == 2  # two distinct designs, not collapsed
    assert len(value_nodes) == 2  # two distinct value nodes despite identical relation+value
    assert {n["label"] for n in entity_nodes} == {"Bi2Te3 single-stage TEC"}
    assert all(n["id"] != n["label"] for n in entity_nodes)  # id is opaque, label is human-readable


def test_unrelated_entity_returns_empty_subgraph(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
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
    subgraph = kg.get_subgraph(["something else entirely"])
    assert subgraph == {"nodes": [], "edges": []}


def test_add_triple_rejects_non_finite_object_value(tmp_path, monkeypatch):
    """The last gate before permanent storage — found via an adversarial
    test where a NaN scalar_result flowed unblocked all the way from
    POST /sciencerag/validate into a queued KG candidate that
    scripts/approve_kg_candidates.py then happily wrote into graph.json as
    a literal (non-standard, RFC-8259-violating) NaN JSON token. API-level
    validation now catches the normal path (ValidateRequest/ReportRequest),
    but add_triple() itself is the one check every write path shares,
    including a hand-assembled --file passed straight to the CLI."""
    _reset_graph_path(tmp_path, monkeypatch)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            kg.add_triple(
                subject="Bi2Te3 single-stage TEC",
                relation="achieves_delta_T_max_K",
                object_value=bad,
                object_unit="K",
                conditions={},
                confidence=0.7,
                run_id="run_bad",
                sources=[],
            )
    assert kg._load_triples() == []


def test_different_conditions_get_different_entity_ids(tmp_path, monkeypatch):
    # Regression for the bug found 2026-08-11: `subject` alone used to be
    # the graph identity, so two different designs (different geometry in
    # `conditions`) sharing the same descriptive subject string collapsed
    # onto one node. entity_id must individuate them.
    _reset_graph_path(tmp_path, monkeypatch)
    a, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    b, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=58.2,
        object_unit="K",
        conditions={"leg_length": 0.5},
        confidence=0.7,
        run_id="run_2",
        sources=[],
    )
    assert a.entity_id != b.entity_id


def test_same_conditions_reuse_same_entity_id(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    a, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    b, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.48,
        object_unit="A",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    assert a.entity_id == b.entity_id


def test_sparse_conditions_with_different_relations_do_not_merge(tmp_path, monkeypatch):
    # The (entity_id, relation) dedup key must keep relation as a real
    # dimension — entity_id alone would collapse these two (same subject,
    # both conditions={}) into one merge/conflict instead of two additions.
    _reset_graph_path(tmp_path, monkeypatch)
    a, status_a = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    b, status_b = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.48,
        object_unit="A",
        conditions={},
        confidence=0.7,
        run_id="run_1",
        sources=[],
    )
    assert status_a == status_b == "added"
    assert a.triple_id != b.triple_id
    assert len(kg._load_triples()) == 2


def test_query_kg_entities_returns_full_entity_not_partial_slice(tmp_path, monkeypatch):
    # Regression for the bug found 2026-08-11: a generic query against a
    # multi-design graph used to return an arbitrary row-based slice that
    # could split one design's data across the cutoff. Once an entity is
    # selected, ALL of its triples must come back together.
    _reset_graph_path(tmp_path, monkeypatch)
    for relation, value, unit in [
        ("achieves_delta_T_max_K", 71.7, "K"),
        ("achieves_optimal_current_A", 6.48, "A"),
        ("achieves_optimal_voltage_V", 0.27, "V"),
    ]:
        kg.add_triple(
            subject="Bi2Te3 single-stage TEC",
            relation=relation,
            object_value=value,
            object_unit=unit,
            conditions={"leg_length": 0.8},
            confidence=0.7,
            run_id="run_1",
            sources=[],
        )
    result = kg.query_kg_entities("Bi2Te3 single-stage TEC achieves_delta_T_max_K")
    assert result.entities_returned == 1
    assert result.total_matching_entities == 1
    assert len(result.groups[0].triples) == 3  # all 3, not just the one matching term-for-term


def test_query_kg_entities_reports_truncation_honestly(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    for i in range(3):
        kg.add_triple(
            subject="Bi2Te3 single-stage TEC",
            relation="achieves_delta_T_max_K",
            object_value=float(i),
            object_unit="K",
            conditions={"leg_length": float(i)},
            confidence=0.7,
            run_id="run_1",
            sources=[],
        )
    result = kg.query_kg_entities("Bi2Te3 single-stage TEC achieves_delta_T_max_K", max_entities=2)
    assert result.total_matching_entities == 3
    assert result.entities_returned == 2
    assert len(result.groups) == 2


def test_pure_chinese_query_matches_relation_description(tmp_path, monkeypatch):
    # Regression for the bug found 2026-08-11: subject/relation/conditions
    # are all English contract identifiers, so a natural Chinese question
    # with no English/digit terms in it (e.g. Hermes asking plainly instead
    # of quoting a parameter name) used to tokenize to an empty set and
    # score 0 against every triple in the graph, even when the triple's own
    # relation_description said exactly what was being asked. Fixed by (1)
    # segmenting CJK text in _tokenize via jieba and (2) including
    # relation_description in the text _render_text scores against.
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.4831,
        object_unit="A",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最优电流",
    )
    hits = kg.query_kg("最优电流是什么")
    assert hits, "pure-Chinese question should match a triple whose relation_description covers it"
    assert hits[0].relevance > 0


def test_chinese_stopwords_do_not_dilute_relevance_below_threshold(tmp_path, monkeypatch):
    # A realistic Hermes-style question has grammar words ("是什么") mixed
    # in with the real content words ("电流"). Those grammar words must not
    # inflate len(query_terms) enough to push a genuine match's relevance
    # below sciencerag.ask.pipeline.MIN_KG_RELEVANCE (0.5).
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.4831,
        object_unit="A",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最优电流",
    )
    hits = kg.query_kg("最优电流是什么")
    assert hits[0].relevance >= 0.5


def test_tokenize_segments_cjk_and_drops_stopwords():
    terms = kg._tokenize("最优电流是什么")
    assert "电流" in terms or "最优电流" in terms
    assert "是" not in terms
    assert "什么" not in terms


def test_render_text_includes_relation_description():
    triple = kg.KGTriple(
        triple_id="t1",
        entity_id="e1",
        entity_type="TECDesign",
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.4831,
        object_unit="A",
        relation_description="最优电流",
        conditions={},
        confidence=0.7,
        run_ids=["run_1"],
        sources=[],
        created_at="2026-08-11T00:00:00+00:00",
    )
    assert "最优电流" in kg._render_text(triple)


def _add_design(subject_leg_length, delta_t, relation_description="最大温差"):
    return kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=delta_t,
        object_unit="K",
        conditions={"leg_length": subject_leg_length},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description=relation_description,
    )


def test_no_ranking_signal_returns_no_ranking(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    _add_design(0.8, 71.7)
    _stub_ranking_classification(monkeypatch, is_ranking=False)
    assert kg.rank_kg_entities("温差是多少") is None


def test_rank_kg_entities_sorts_by_value_max(tmp_path, monkeypatch):
    # Regression for the 2026-08-11 first-principles finding: a superlative
    # word in a query used to be treated as a plain keyword (matched
    # against relation_description text) with no actual comparison ever
    # computed — every design scored ~equal relevance and the LLM was left
    # to guess which one was "best" from an unordered dump. This must
    # genuinely sort by object_value, not by text overlap.
    _reset_graph_path(tmp_path, monkeypatch)
    low, _ = _add_design(0.5, 58.2)
    high, _ = _add_design(0.8, 81.5)
    mid, _ = _add_design(0.65, 65.4)
    _stub_ranking_classification(
        monkeypatch, is_ranking=True, relation="achieves_delta_T_max_K", direction="max"
    )
    result = kg.rank_kg_entities("哪个设计的最大温差最高", top_k=2)
    assert result is not None
    assert result.relation == "achieves_delta_T_max_K"
    assert result.direction == "max"
    assert result.total_candidates == 3
    assert [e.entity_id for e in result.ranked] == [high.entity_id, mid.entity_id]
    assert result.ranked[0].value == 81.5


def test_rank_kg_entities_sorts_by_value_min(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    low, _ = _add_design(0.5, 58.2)
    high, _ = _add_design(0.8, 81.5)
    _stub_ranking_classification(
        monkeypatch, is_ranking=True, relation="achieves_delta_T_max_K", direction="min"
    )
    result = kg.rank_kg_entities("最低温差是多少", top_k=1)
    assert result is not None
    assert result.direction == "min"
    assert result.ranked[0].entity_id == low.entity_id
    assert result.ranked[0].value == 58.2


def test_rank_kg_entities_relation_from_classification(tmp_path, monkeypatch):
    # The field comes straight from the classifier's `relation`, not from
    # any text-overlap heuristic — confirms rank_kg_entities trusts the
    # classification's field choice directly, regardless of what the
    # stored relation_description happens to say.
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_optimal_current_A",
        object_value=6.4831,
        object_unit="A",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[],
        relation_description="最优电流",
    )
    _stub_ranking_classification(
        monkeypatch, is_ranking=True, relation="achieves_optimal_current_A", direction="max"
    )
    result = kg.rank_kg_entities("最高电流是多少")
    assert result is not None
    assert result.relation == "achieves_optimal_current_A"


def test_rank_kg_entities_rejects_relation_outside_candidate_set(tmp_path, monkeypatch):
    # A classification naming a relation that doesn't exist in the graph
    # (hallucination, or a stale answer) must degrade to no-ranking, not
    # raise or silently rank an empty/wrong set.
    _reset_graph_path(tmp_path, monkeypatch)
    _add_design(0.8, 71.7)
    _stub_ranking_classification(
        monkeypatch, is_ranking=True, relation="does_not_exist", direction="max"
    )
    assert kg.rank_kg_entities("最高温差是多少") is None


def test_rank_kg_entities_empty_graph_returns_none(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    assert kg.rank_kg_entities("最优电流是多少") is None


def test_classify_ranking_query_best_effort_on_llm_failure(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    _add_design(0.8, 71.7)

    def fake_completion(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(kg.litellm, "completion", fake_completion)
    assert kg.rank_kg_entities("最高温差是多少") is None


def test_backfill_entity_fields_migrates_legacy_records(tmp_path, monkeypatch):
    path = _reset_graph_path(tmp_path, monkeypatch)
    # Write records in the pre-migration shape directly (no entity_id/
    # entity_type), matching what real legacy data/kg/graph.json looked
    # like before this migration existed.
    path.write_text(
        json.dumps(
            [
                {
                    "triple_id": "kg_legacy1",
                    "subject": "Bi2Te3 single-stage TEC",
                    "relation": "achieves_delta_T_max_K",
                    "object_value": 71.7,
                    "object_unit": "K",
                    "conditions": {"leg_length": 0.8},
                    "confidence": 0.7,
                    "run_ids": ["run_1"],
                    "sources": [],
                    "conflicts_with": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )
    updated = kg.backfill_entity_fields()
    assert updated == 1
    triples = kg._load_triples()
    assert len(triples) == 1
    assert triples[0].entity_id == kg.compute_entity_id(
        "Bi2Te3 single-stage TEC", {"leg_length": 0.8}
    )
    assert triples[0].entity_type == kg.DEFAULT_ENTITY_TYPE
    # idempotent — running it again on already-migrated data changes nothing
    assert kg.backfill_entity_fields() == 0


def test_add_triple_rejects_both_or_neither_object_kind(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        kg.add_triple(
            subject="s", relation="r", conditions={}, confidence=0.7, run_id="run_1", sources=[]
        )
    with pytest.raises(ValueError, match="exactly one"):
        kg.add_triple(
            subject="s",
            relation="r",
            conditions={},
            confidence=0.7,
            run_id="run_1",
            sources=[],
            object_value=1.0,
            object_entity_id="tec_other",
        )


def test_link_triple_renders_entity_to_entity_edge(tmp_path, monkeypatch):
    # This is the actual fix for "the graph is a bunch of disconnected
    # star clusters" (found 2026-08-11): a link triple's target must be a
    # real, shared entity node — not a private per-triple value node — so
    # two different subject entities that both link to it show up
    # connected in the rendered graph.
    _reset_graph_path(tmp_path, monkeypatch)
    material_id = kg.compute_entity_id("Bi2Te3", {})
    design_a, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="SIMULATION_USES_MATERIAL",
        conditions={"leg_length": 0.8},
        confidence=1.0,
        run_id="run_1",
        sources=[],
        object_entity_id=material_id,
        object_entity_label="Bi2Te3",
        object_entity_type="Material",
        entity_type="SimulationRun",
    )
    design_b, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="SIMULATION_USES_MATERIAL",
        conditions={"leg_length": 0.5},
        confidence=1.0,
        run_id="run_2",
        sources=[],
        object_entity_id=material_id,
        object_entity_label="Bi2Te3",
        object_entity_type="Material",
        entity_type="SimulationRun",
    )
    assert design_a.object_entity_id == design_b.object_entity_id == material_id

    subgraph = kg.full_graph()
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert material_id in node_ids
    assert all(n["kind"] == "entity" for n in subgraph["nodes"])  # no value node at all here
    # both runs' edges point at the SAME shared material node
    material_edges = [e for e in subgraph["edges"] if e["target"] == material_id]
    assert len(material_edges) == 2
    assert {e["source"] for e in material_edges} == {design_a.entity_id, design_b.entity_id}


def test_link_triple_merges_on_same_target_conflicts_on_different(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    first, status1 = kg.add_triple(
        subject="run A",
        relation="SIMULATION_USES_MATERIAL",
        conditions={},
        confidence=1.0,
        run_id="run_1",
        sources=[],
        object_entity_id="tec_material_a",
        object_entity_label="Material A",
        object_entity_type="Material",
    )
    same_target, status2 = kg.add_triple(
        subject="run A",
        relation="SIMULATION_USES_MATERIAL",
        conditions={},
        confidence=1.0,
        run_id="run_2",
        sources=[],
        object_entity_id="tec_material_a",
        object_entity_label="Material A",
        object_entity_type="Material",
    )
    assert status1 == "added"
    assert status2 == "merged"
    assert same_target.triple_id == first.triple_id

    different_target, status3 = kg.add_triple(
        subject="run A",
        relation="SIMULATION_USES_MATERIAL",
        conditions={},
        confidence=1.0,
        run_id="run_3",
        sources=[],
        object_entity_id="tec_material_b",
        object_entity_label="Material B",
        object_entity_type="Material",
    )
    assert status3 == "conflict"
    assert different_target.conflicts_with == first.triple_id


def test_merging_the_same_source_twice_does_not_duplicate_it(tmp_path, monkeypatch):
    # Regression for a real bug found via adversarial review (2026-08-14/15),
    # confirmed already present in production data/kg/graph.json: run_ids
    # dedupes on merge, but sources never did — re-merging the exact same
    # source (a retried request, the same demo script run twice) silently
    # accumulated exact-duplicate KGSource entries forever.
    _reset_graph_path(tmp_path, monkeypatch)
    source = kg.KGSource(type="run", run_id="run_1")
    first, _ = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[source],
    )
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[source],  # the exact same source, submitted again
    )
    assert status == "merged"
    assert second.sources == [source]  # not [source, source]


def test_merging_a_genuinely_new_source_still_appends(tmp_path, monkeypatch):
    _reset_graph_path(tmp_path, monkeypatch)
    kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_1",
        sources=[kg.KGSource(type="run", run_id="run_1")],
    )
    second, status = kg.add_triple(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=71.7,
        object_unit="K",
        conditions={"leg_length": 0.8},
        confidence=0.7,
        run_id="run_2",
        sources=[kg.KGSource(type="run", run_id="run_2")],  # a genuinely different source
    )
    assert status == "merged"
    assert len(second.sources) == 2


def test_concurrent_add_triple_does_not_lose_writes(tmp_path, monkeypatch):
    """Regression test for a real, reproduced bug: 20 concurrent
    add_triple() calls against the same graph, before the fix, resulted in
    only 6 surviving triples (14 silently lost to an unlocked
    read-modify-write race) plus several JSON parse errors from readers
    catching a writer's write_text() mid-write. The fix adds an
    fcntl.flock around the whole load-mutate-save cycle plus an atomic
    (write-temp-then-rename) save, so every concurrent addition must
    survive and no reader should ever see a partially-written file."""
    _reset_graph_path(tmp_path, monkeypatch)
    n = 20
    errors: list[tuple[int, str]] = []

    def add_one(i: int) -> None:
        try:
            kg.add_triple(
                subject="Bi2Te3 single-stage TEC",
                relation=f"race_test_relation_{i}",
                object_value=float(i),
                object_unit="K",
                conditions={},
                confidence=0.7,
                run_id=f"race_run_{i}",
                sources=[],
            )
        except Exception as e:  # noqa: BLE001 - captured for the assertion below
            errors.append((i, str(e)))

    threads = [threading.Thread(target=add_one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(kg._load_triples()) == n
