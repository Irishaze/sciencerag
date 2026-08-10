"""Tests for the real NetworkX/JSON-backed graph store (spec §3.2/§4.4/§6,
M5). Points GRAPH_PATH at a tmp file per test so these don't touch/depend
on the real data/kg/graph.json.
"""

import threading

import pytest

from sciencerag.priors import kg


def _reset_graph_path(tmp_path, monkeypatch):
    path = tmp_path / "graph.json"
    monkeypatch.setattr(kg, "GRAPH_PATH", path)
    return path


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
