"""Tests for scripts/approve_kg_candidates.py's --pending queue flow."""

import importlib.util
import sys
from pathlib import Path

import pytest

from sciencerag.common import audit
from sciencerag.priors import kg
from sciencerag.validate import kg_candidate_store

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "approve_kg_candidates.py"
_spec = importlib.util.spec_from_file_location("approve_kg_candidates_script", SCRIPT_PATH)
approve_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(approve_script)


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(kg, "GRAPH_PATH", tmp_path / "graph.json")
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", tmp_path / "audit.jsonl")


def _queue_one_candidate(run_id="run_test", relation="achieves_delta_T_max_K"):
    from sciencerag.validate.models import KGCandidate

    candidate = KGCandidate.model_validate(
        {
            "subject": "Bi2Te3 single-stage TEC",
            "relation": relation,
            "object_value": 70.0,
            "object_unit": "K",
            "conditions": {},
            "confidence": 0.7,
            "run_id": run_id,
            "dedup_status": "new",
            "entity_type": "TECDesign",
        }
    )
    kg_candidate_store.store_pending_candidates(run_id, [candidate])
    return kg_candidate_store.list_pending()[0]["stem"]


def test_list_pending_reports_queued_batch(monkeypatch, capsys):
    _queue_one_candidate()
    monkeypatch.setattr(sys, "argv", ["approve_kg_candidates.py", "--list-pending"])
    approve_script.main()
    out = capsys.readouterr().out
    assert "run_test_" in out
    assert "1 candidate(s)" in out


def test_approve_all_from_pending_writes_triple_and_archives(monkeypatch, capsys):
    stem = _queue_one_candidate()
    monkeypatch.setattr(sys, "argv", ["approve_kg_candidates.py", "--pending", stem, "--approve-all"])
    approve_script.main()

    out = capsys.readouterr().out
    assert "added: triple_id=" in out

    # Approved batch should no longer show up in the pending queue...
    assert kg_candidate_store.list_pending() == []
    # ...but should be archived, not deleted.
    assert (kg_candidate_store.ARCHIVE_DIR / f"{stem}.json").exists()

    triples = kg._load_triples()
    assert len(triples) == 1
    assert triples[0].relation == "achieves_delta_T_max_K"


def test_preview_only_does_not_write_or_archive(monkeypatch, capsys):
    stem = _queue_one_candidate()
    monkeypatch.setattr(sys, "argv", ["approve_kg_candidates.py", "--pending", stem])
    approve_script.main()

    assert kg._load_triples() == []
    assert len(kg_candidate_store.list_pending()) == 1
