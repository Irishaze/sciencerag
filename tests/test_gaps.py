"""Tests for the gaps heuristic (sciencerag/priors/retrieval.py).

Pure-function tests against constructed Prior objects — no PaperQA2/API calls.
"""

from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.priors.retrieval import _build_gaps, _split_by_confidence


def _prior(confidence: float, notes: str | None = None) -> Prior:
    return Prior(
        prior_id="pr_test",
        kind="parameter_range",
        field="general_finding",
        value={"summary": "x"},
        confidence=confidence,
        sources=[SourcePaper(doi="10.0000/test", span="pages 1-2")],
        notes=notes,
    )


def test_split_by_confidence_separates_strong_and_weak():
    priors = [_prior(0.9), _prior(0.5), _prior(0.4), _prior(0.1)]
    strong, weak = _split_by_confidence(priors)
    assert [p.confidence for p in strong] == [0.9, 0.5]
    assert [p.confidence for p in weak] == [0.4, 0.1]


def test_build_gaps_empty_when_no_weak_priors():
    assert _build_gaps([], total_hits=5) == []


def test_build_gaps_notes_excluded_count_and_papers():
    weak = [_prior(0.2, notes="Paper A"), _prior(0.3, notes="Paper B")]
    gaps = _build_gaps(weak, total_hits=5)
    assert len(gaps) == 1
    assert "2 evidence context" in gaps[0]
    assert "Paper A" in gaps[0]
    assert "Paper B" in gaps[0]


def test_build_gaps_reports_zero_hits_case():
    gaps = _build_gaps([], total_hits=0)
    assert len(gaps) == 1
    assert "no relevant evidence" in gaps[0]
