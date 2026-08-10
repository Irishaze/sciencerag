"""Unit tests for the pending-candidates queue (sciencerag/validate/kg_candidate_store.py)."""

import pytest

from sciencerag.validate import kg_candidate_store
from sciencerag.validate.models import KGCandidate


@pytest.fixture(autouse=True)
def _tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(kg_candidate_store, "PENDING_DIR", tmp_path / "pending")
    monkeypatch.setattr(kg_candidate_store, "ARCHIVE_DIR", tmp_path / "archive")


def _candidate(**overrides) -> KGCandidate:
    base = dict(
        subject="Bi2Te3 single-stage TEC",
        relation="achieves_delta_T_max_K",
        object_value=70.0,
        object_unit="K",
        conditions={},
        confidence=0.7,
        run_id="run_test",
        dedup_status="new",
    )
    base.update(overrides)
    return KGCandidate.model_validate(base)


def test_empty_candidates_are_not_queued():
    assert kg_candidate_store.store_pending_candidates("run_test", []) is None
    assert kg_candidate_store.list_pending() == []


def test_store_then_list_then_load_round_trips():
    candidates = [_candidate(), _candidate(relation="achieves_cop")]
    path = kg_candidate_store.store_pending_candidates("run_test", candidates)
    assert path is not None and path.exists()

    pending = kg_candidate_store.list_pending()
    assert len(pending) == 1
    assert pending[0]["count"] == 2
    assert pending[0]["stem"].startswith("run_test_")

    loaded = kg_candidate_store.load_pending(pending[0]["stem"])
    assert [c.relation for c in loaded] == ["achieves_delta_T_max_K", "achieves_cop"]


def test_archive_removes_from_pending_list():
    kg_candidate_store.store_pending_candidates("run_test", [_candidate()])
    stem = kg_candidate_store.list_pending()[0]["stem"]

    kg_candidate_store.archive_pending(stem)

    assert kg_candidate_store.list_pending() == []
    assert (kg_candidate_store.ARCHIVE_DIR / f"{stem}.json").exists()
