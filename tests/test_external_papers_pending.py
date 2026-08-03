"""Tests for the pending-external-papers queue (spec §3.5 step 1) that
scripts/approve_external_papers.py reads."""

import pytest

from sciencerag.priors import external_retrieval
from sciencerag.priors.external_retrieval import ExternalPaper, record_pending_papers


@pytest.fixture(autouse=True)
def _tmp_pending_path(tmp_path, monkeypatch):
    monkeypatch.setattr(external_retrieval, "PENDING_PATH", tmp_path / "pending.json")


def _paper(doi: str = "10.1/x") -> ExternalPaper:
    return ExternalPaper(title="T", abstract="A", doi=doi, year=2024, semantic_scholar_id="s1")


def test_new_paper_recorded_with_hit_count_one():
    record_pending_papers([_paper()])
    pending = external_retrieval._load_pending()
    assert pending["10.1/x"].hit_count == 1


def test_repeated_paper_increments_hit_count():
    record_pending_papers([_paper()])
    record_pending_papers([_paper()])
    record_pending_papers([_paper()])
    pending = external_retrieval._load_pending()
    assert pending["10.1/x"].hit_count == 3


def test_empty_list_is_a_no_op():
    record_pending_papers([])
    assert external_retrieval._load_pending() == {}
