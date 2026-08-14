"""Tests for scripts/seed_kg_from_corpus.py's per-query fault isolation.

Adversarial test, confirmed live before the fix: build_priors_response is a
real network + LLM call per query (seconds to ~2 minutes each, per the
script's own docstring). An unhandled failure on query N used to crash the
whole script and discard every candidate already extracted from queries
1..N-1, forcing a full expensive re-run to recover work that had already
been earned.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from sciencerag.priors.models import Coverage, Prior, PriorsResponse, ScalingRelationshipValue, SourcePaper

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "seed_kg_from_corpus.py"
_spec = importlib.util.spec_from_file_location("seed_kg_from_corpus_script", SCRIPT_PATH)
seed_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_script)


def _scaling_prior(prior_id: str) -> Prior:
    return Prior(
        prior_id=prior_id,
        kind="scaling_relationship",
        related_fields=["leg_length", "leg_width"],
        value=ScalingRelationshipValue(x="leg_length", y="leg_width", direction="positive"),
        confidence=0.6,
        sources=[SourcePaper(doi="10.1234/demo")],
    )


def test_one_failing_query_does_not_discard_earlier_candidates(monkeypatch, tmp_path, capsys):
    from sciencerag.priors import kg
    from sciencerag.validate import kg_candidate_store

    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(seed_script, "kg_candidate_store", kg_candidate_store)

    call_count = {"n": 0}

    def flaky_build(query, allow_external, max_priors):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated LLM API timeout")
        return PriorsResponse(
            priors=[_scaling_prior(f"pr_{call_count['n']}")],
            coverage=Coverage(gaps=[], internal_hits=1, external_hits=0),
            trace_id="tr_test",
        ), []

    monkeypatch.setattr(seed_script, "build_priors_response", flaky_build)
    monkeypatch.setattr(
        sys, "argv", ["seed_kg_from_corpus.py", "--queries", "q1", "q2 (fails)", "q3"]
    )

    seed_script.main()  # must not raise

    assert call_count["n"] == 3  # all three queries attempted, not aborted after the failure
    pending = kg_candidate_store.list_pending()
    assert len(pending) == 1
    # queries 1 and 3 each produced one scaling_relationship candidate;
    # query 2's failure cost only itself, not the other two.
    assert pending[0]["count"] == 2

    out = capsys.readouterr().out
    assert "simulated LLM API timeout" in out
    assert "1 queries failed and were skipped" in out or "1 query failed and were skipped" in out


def test_all_queries_failing_reports_nothing_queued(monkeypatch, tmp_path, capsys):
    from sciencerag.priors import kg
    from sciencerag.validate import kg_candidate_store

    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(seed_script, "kg_candidate_store", kg_candidate_store)

    def always_fails(query, allow_external, max_priors):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(seed_script, "build_priors_response", always_fails)
    monkeypatch.setattr(sys, "argv", ["seed_kg_from_corpus.py", "--queries", "q1"])

    seed_script.main()  # must not raise

    assert kg_candidate_store.list_pending() == []
    assert "no candidates produced" in capsys.readouterr().out
