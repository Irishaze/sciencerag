"""Tests for the gaps heuristic (sciencerag/priors/retrieval.py).

Pure-function tests against constructed Prior objects — no PaperQA2/API calls.
`_build_geometry_gaps` takes a pre-computed `relevance_matched_params` set
(the caller resolves this via `_match_params_to_evidence`, a real LLM call —
see test_param_matching.py for that side); here we just supply the set
directly to keep these tests API-free.

Confidence no longer gates what counts as "extracted" (2026-08-05 — see
extract.py's module docstring): the only "extracted but not returned"
reason left is the semantic support judge's REVIEW verdict, or getting cut
by the max_priors cap. `_prior`'s `confidence` param below is now purely a
ranking key for `_cap_priors`, not a pass/fail signal.
"""

from sciencerag.priors.contract import GEOMETRY_FREE_NAMES
from sciencerag.priors.extract import ReviewedPrior
from sciencerag.priors.models import Prior, SourcePaper
from sciencerag.priors.retrieval import (
    _build_gaps,
    _build_geometry_gaps,
    _build_max_priors_gap,
    _cap_priors,
)


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


def _reviewed(reason: str = "plausible domain inference", **prior_kwargs) -> ReviewedPrior:
    return ReviewedPrior(prior=_prior(0.6, **prior_kwargs), reason=reason)


def test_build_gaps_empty_when_no_review_priors():
    assert _build_gaps([], total_hits=5) == []


def test_build_gaps_notes_excluded_count_papers_and_reasons():
    """The gaps message describes REVIEW-excluded PRIORS, not raw evidence
    contexts — a prior may already merge multiple evidence snippets (see
    extract.py), so "N evidence contexts" would misdescribe what actually
    got excluded. Regression test for that exact wording bug."""
    reviewed = [
        _reviewed(doi="10.1111/a", reason="aspect ratio mapping"),
        _reviewed(doi="10.2222/b", reason="fill factor mapping"),
    ]
    gaps = _build_gaps(reviewed, total_hits=5)
    assert len(gaps) == 1
    assert "2 prior" in gaps[0]
    assert "evidence context" not in gaps[0]
    assert "10.1111/a" in gaps[0]
    assert "10.2222/b" in gaps[0]
    assert "aspect ratio mapping" in gaps[0]
    assert "fill factor mapping" in gaps[0]


def test_build_gaps_uses_doi_not_llm_notes():
    """Prior.notes is the LLM's own clarifying note when it provides one —
    NOT reliably a paper title (see extract.py's _to_prior fallback logic).
    Using it to say "which paper" would sometimes show LLM commentary
    instead of a source. Regression test for that exact bug, found via
    manual review of a real gaps message showing LLM notes text instead of
    a paper reference."""
    reviewed = [
        _reviewed(notes="Based on prior work cited by E9 (Han et al.)", doi="10.3333/c")
    ]
    gaps = _build_gaps(reviewed, total_hits=5)
    assert "10.3333/c" in gaps[0]
    assert "Based on prior work" not in gaps[0]


def test_build_gaps_reports_zero_hits_case():
    gaps = _build_gaps([], total_hits=0)
    assert len(gaps) == 1
    assert "no relevant evidence" in gaps[0]


# -- contract-based coverage gaps (spec §3.6) ---------


def test_build_geometry_gaps_reports_all_12_when_nothing_extracted():
    gaps = _build_geometry_gaps(
        all_kept_priors=[], returned_priors=[], relevance_matched_params=set()
    )
    assert len(gaps) == len(GEOMETRY_FREE_NAMES)
    assert all("未检索到" in g for g in gaps)


def test_build_geometry_gaps_excludes_params_covered_by_a_returned_prior():
    returned = [_prior(0.9, field="leg_length")]
    gaps = _build_geometry_gaps(
        all_kept_priors=returned, returned_priors=returned, relevance_matched_params=set()
    )
    assert not any("leg_length" in g for g in gaps)
    assert len(gaps) == len(GEOMETRY_FREE_NAMES) - 1


def test_build_geometry_gaps_excludes_params_covered_via_related_fields():
    returned = [_prior(0.9, field=None, related_fields=["leg_length", "leg_width"])]
    gaps = _build_geometry_gaps(
        all_kept_priors=returned, returned_priors=returned, relevance_matched_params=set()
    )
    assert not any("leg_length" in g for g in gaps)
    assert not any("leg_width" in g for g in gaps)


def test_build_geometry_gaps_distinguishes_not_returned_from_uncovered():
    """A prior that was extracted (passed numeric+semantic checks) but
    didn't make it into `returned_priors` (REVIEW-excluded or max_priors-
    capped — the caller decides which via `all_kept_priors`'s membership)
    is a different gap reason than "nothing extracted at all"."""
    drafted_not_returned = _prior(0.1, field="leg_length")
    gaps = _build_geometry_gaps(
        all_kept_priors=[drafted_not_returned],
        returned_priors=[],
        relevance_matched_params=set(),
    )
    leg_length_gap = next(g for g in gaps if "leg_length" in g)
    assert "未进入最终结果" in leg_length_gap
    other_gap = next(g for g in gaps if "leg_width" in g)
    assert "未检索到" in other_gap


# -- 3rd tier: relevance-filtered evidence (spec §3.7's 3-way split) -------
# `relevance_matched_params` is the caller's already-resolved answer (from
# _match_params_to_evidence, a real LLM call) for which params at least one
# below-threshold evidence snippet discusses — these tests just check the
# tier-priority logic that consumes that set, not how it's computed.


def test_build_geometry_gaps_reports_relevance_filtered_when_param_matched():
    gaps = _build_geometry_gaps(
        all_kept_priors=[], returned_priors=[], relevance_matched_params={"leg_length"}
    )
    leg_length_gap = next(g for g in gaps if "leg_length" in g)
    assert "相关性不足" in leg_length_gap
    assert "未进入最终结果" not in leg_length_gap


def test_build_geometry_gaps_falls_back_to_unretrieved_when_no_param_matched():
    gaps = _build_geometry_gaps(
        all_kept_priors=[], returned_priors=[], relevance_matched_params=set()
    )
    leg_length_gap = next(g for g in gaps if "leg_length" in g)
    assert "未检索到" in leg_length_gap


def test_build_geometry_gaps_not_returned_tier_takes_priority_over_relevance_tier():
    """A param that has BOTH a drafted-but-not-returned prior AND matching
    below-threshold evidence should report the drafted-tier reason — the
    prior in hand is more specific/actionable than an evidence guess."""
    drafted_not_returned = _prior(0.1, field="leg_length")
    gaps = _build_geometry_gaps(
        all_kept_priors=[drafted_not_returned],
        returned_priors=[],
        relevance_matched_params={"leg_length"},
    )
    leg_length_gap = next(g for g in gaps if "leg_length" in g)
    assert "未进入最终结果" in leg_length_gap


# -- max_priors cap ---------


def test_cap_priors_keeps_highest_confidence_when_under_cap():
    priors = [_prior(0.6), _prior(0.9), _prior(0.7)]
    returned, truncated = _cap_priors(priors, max_priors=5)
    assert [p.confidence for p in returned] == [0.9, 0.7, 0.6]
    assert truncated == []


def test_cap_priors_truncates_lowest_confidence_first():
    priors = [_prior(0.6), _prior(0.9), _prior(0.7), _prior(0.55)]
    returned, truncated = _cap_priors(priors, max_priors=2)
    assert [p.confidence for p in returned] == [0.9, 0.7]
    assert [p.confidence for p in truncated] == [0.6, 0.55]


# -- max_priors cap gaps ---------


def test_build_max_priors_gap_empty_when_nothing_truncated():
    assert _build_max_priors_gap([], max_priors=5) == []


def test_build_max_priors_gap_notes_excluded_count_and_papers():
    truncated = [_prior(0.6, doi="10.1111/a"), _prior(0.55, doi="10.2222/b")]
    gaps = _build_max_priors_gap(truncated, max_priors=3)
    assert len(gaps) == 1
    assert "2 additional prior" in gaps[0]
    assert "max_priors=3" in gaps[0]
    assert "10.1111/a" in gaps[0]
    assert "10.2222/b" in gaps[0]
