"""Tests for the gaps heuristic (sciencerag/priors/retrieval.py).

Pure-function tests against constructed Prior objects — no PaperQA2/API calls.
"""

from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.priors.retrieval import _build_gaps, _split_by_confidence


def _prior(confidence: float, notes: str | None = None, doi: str = "10.0000/test") -> Prior:
    return Prior(
        prior_id="pr_test",
        kind="parameter_range",
        field="general_finding",
        value={"summary": "x"},
        confidence=confidence,
        sources=[SourcePaper(doi=doi, span="pages 1-2")],
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
    """The gaps message describes low-confidence PRIORS, not raw evidence
    contexts — a prior may already merge multiple evidence snippets (see
    extract.py), so "N evidence contexts" would misdescribe what actually
    got excluded. Regression test for that exact wording bug."""
    weak = [_prior(0.2, doi="10.1111/a"), _prior(0.3, doi="10.2222/b")]
    gaps = _build_gaps(weak, total_hits=5)
    assert len(gaps) == 1
    assert "2 low-confidence prior" in gaps[0]
    assert "evidence context" not in gaps[0]
    assert "10.1111/a" in gaps[0]
    assert "10.2222/b" in gaps[0]


def test_build_gaps_uses_doi_not_llm_notes():
    """Prior.notes is the LLM's own clarifying note when it provides one —
    NOT reliably a paper title (see extract.py's _to_prior fallback logic).
    Using it to say "which paper" would sometimes show LLM commentary
    instead of a source. Regression test for that exact bug, found via
    manual review of a real gaps message showing LLM notes text instead of
    a paper reference."""
    weak = [_prior(0.2, notes="Based on prior work cited by E9 (Han et al.)", doi="10.3333/c")]
    gaps = _build_gaps(weak, total_hits=5)
    assert "10.3333/c" in gaps[0]
    assert "Based on prior work" not in gaps[0]


def test_build_gaps_reports_zero_hits_case():
    gaps = _build_gaps([], total_hits=0)
    assert len(gaps) == 1
    assert "no relevant evidence" in gaps[0]
