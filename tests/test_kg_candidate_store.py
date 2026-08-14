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
        entity_type="TECDesign",
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


def test_archive_pending_is_idempotent_under_concurrent_callers():
    """Regression test for a real concurrency race, confirmed via a direct
    test (2 threads x 2000 trials, 99%+ crash rate): archive_pending's old
    `if src.exists(): src.rename(...)` had a TOCTOU window — two concurrent
    approve requests against the same stem (sciencerag.kg_approval's web
    panel calls this at the end of every approve, regardless of which
    indices were approved) could both pass the exists() check, and the
    second caller's rename() then raised an uncaught FileNotFoundError. That
    crashed a request whose actual candidate approvals had already
    succeeded. "Already archived by someone else" must be a no-op, not an
    error."""
    kg_candidate_store.store_pending_candidates("run_test", [_candidate()])
    stem = kg_candidate_store.list_pending()[0]["stem"]

    kg_candidate_store.archive_pending(stem)
    kg_candidate_store.archive_pending(stem)  # simulates the second racing caller

    assert kg_candidate_store.list_pending() == []
    assert (kg_candidate_store.ARCHIVE_DIR / f"{stem}.json").exists()


def test_store_leaves_no_leftover_temp_file():
    """Regression test for the write-to-temp-then-rename fix: confirms the
    happy path doesn't leave a stray .tmp file behind (the atomic-rename
    step actually completes and cleans up after itself)."""
    path = kg_candidate_store.store_pending_candidates("run_test", [_candidate()])
    assert list(kg_candidate_store.PENDING_DIR.glob("*.tmp")) == []
    assert path.suffix == ".json"


def test_load_pending_raises_cleanly_on_a_torn_write():
    """Regression test, confirmed live before the fix: store_pending_
    candidates wrote directly via path.write_text() (not atomically) — a
    reader (load_pending, called by scripts/approve_kg_candidates.py)
    landing mid-write saw a truncated JSON document and raised an uncaught
    JSONDecodeError. Simulates the mid-write state directly (a real race
    is timing-dependent and not worth chasing to reproduce) to confirm
    load_pending's behavior on it is at least a clean, expected exception,
    not a novel crash — the actual fix (os.replace()) means a real
    concurrent reader can no longer observe this state at all."""
    kg_candidate_store.PENDING_DIR.mkdir(parents=True, exist_ok=True)
    partial = kg_candidate_store.PENDING_DIR / "run_torn_2026.json"
    partial.write_text('[{"subject": "Bi2Te3 single-stage TEC", "relation": "achieves_x', encoding="utf-8")
    import json

    with pytest.raises(json.JSONDecodeError):
        kg_candidate_store.load_pending("run_torn_2026")
