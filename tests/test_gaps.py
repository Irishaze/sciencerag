"""Tests for the gaps heuristic (sciencerag/priors/retrieval.py).

Pure-function tests against constructed Prior objects — no PaperQA2/API calls.
"""

from sciencerag.priors.contract import GEOMETRY_FREE_NAMES
from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.priors.retrieval import _build_gaps, _build_geometry_gaps, _split_by_confidence


def _prior(
    confidence: float,
    notes: str | None = None,
    doi: str = "10.0000/test",
    field: str | None = "general_finding",
    related_fields: list[str] | None = None,
) -> Prior:
    # related_fields (no single `field`) only makes sense for a relationship
    # kind — parameter_range's value.field_name must match the single
    # `field` (see models.py's cross-check), so switch shape accordingly.
    if field is None and related_fields:
        kind = "scaling_relationship"
        value = {"x": related_fields[0], "y": related_fields[1], "direction": "unknown"}
    else:
        kind = "parameter_range"
        value = {"field_name": field or "x", "typical": 1.0, "unit": "mm"}
    return Prior(
        prior_id="pr_test",
        kind=kind,
        field=field,
        related_fields=related_fields or [],
        value=value,
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


# -- contract-based coverage gaps (spec §3.6) ---------


def test_build_geometry_gaps_reports_all_12_when_nothing_extracted():
    gaps = _build_geometry_gaps(all_priors=[], strong_priors=[])
    assert len(gaps) == len(GEOMETRY_FREE_NAMES)
    assert all("未检索到" in g for g in gaps)


def test_build_geometry_gaps_excludes_params_covered_by_a_strong_prior():
    strong = [_prior(0.9, field="leg_length")]
    gaps = _build_geometry_gaps(all_priors=strong, strong_priors=strong)
    assert not any("leg_length" in g for g in gaps)
    assert len(gaps) == len(GEOMETRY_FREE_NAMES) - 1


def test_build_geometry_gaps_excludes_params_covered_via_related_fields():
    strong = [_prior(0.9, field=None, related_fields=["leg_length", "leg_width"])]
    gaps = _build_geometry_gaps(all_priors=strong, strong_priors=strong)
    assert not any("leg_length" in g for g in gaps)
    assert not any("leg_width" in g for g in gaps)


def test_build_geometry_gaps_distinguishes_low_confidence_from_uncovered():
    weak = _prior(0.1, field="leg_length")
    gaps = _build_geometry_gaps(all_priors=[weak], strong_priors=[])
    leg_length_gap = next(g for g in gaps if "leg_length" in g)
    assert "置信度不足" in leg_length_gap
    other_gap = next(g for g in gaps if "leg_width" in g)
    assert "未检索到" in other_gap
