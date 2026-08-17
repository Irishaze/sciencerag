"""Tests for retrieval._reconcile_kg_literature_duplicates — a real gap
found 2026-08-17: a value seeded into the KG from a past extraction run and
a value THIS query's live literature search drafts fresh are two
independent paths that never checked each other, so a reader had no way to
tell "same paper, cited twice" from "two different papers that
coincidentally round to a similar number" (real example: 7.0mm KG seed vs
7.0mm fresh draft, where the seed's own source turned out to carry no DOI
at all).
"""

from sciencerag.priors.models import Prior, SourceKGTriple, SourcePaper
from sciencerag.priors.retrieval import _reconcile_kg_literature_duplicates


def _parameter_range_prior(
    *, prior_id: str, field: str, typical: float, confidence: float, doi: str | None, notes: str | None = None
) -> Prior:
    sources = [SourcePaper(doi=doi, span="p.1")] if doi else [SourceKGTriple(triple_id="kg_x")]
    return Prior(
        prior_id=prior_id,
        kind="parameter_range",
        field=field,
        value={"field_name": field, "typical": typical, "unit": "mm"},
        confidence=confidence,
        sources=sources,
        notes=notes,
    )


def test_same_doi_and_agreeing_value_collapses_into_the_kg_prior():
    kg_prior = _parameter_range_prior(
        prior_id="pr_kg_seed", field="leg_length", typical=2.75, confidence=0.42, doi="10.1/real"
    )
    fresh_prior = _parameter_range_prior(
        prior_id="pr_E2_leg_length", field="leg_length", typical=2.76, confidence=0.6, doi="10.1/real"
    )

    remaining = _reconcile_kg_literature_duplicates([kg_prior], [fresh_prior])

    assert remaining == []  # the fresh duplicate is dropped
    assert kg_prior.confidence == 0.6  # upgraded — fresh had the higher confidence
    assert "同一 DOI" in kg_prior.notes
    assert "10.1/real" in kg_prior.notes


def test_same_doi_does_not_downgrade_confidence():
    kg_prior = _parameter_range_prior(
        prior_id="pr_kg_seed", field="leg_length", typical=2.75, confidence=0.8, doi="10.1/real"
    )
    fresh_prior = _parameter_range_prior(
        prior_id="pr_E2_leg_length", field="leg_length", typical=2.75, confidence=0.4, doi="10.1/real"
    )

    _reconcile_kg_literature_duplicates([kg_prior], [fresh_prior])

    assert kg_prior.confidence == 0.8  # unchanged, not downgraded


def test_missing_doi_on_either_side_is_left_alone_not_guessed_as_duplicate():
    # This is the real bug: 7.0mm KG seed (no DOI — see kg_approval.py) vs
    # 7.0mm fresh draft. Matching VALUES alone must never be enough to merge.
    kg_prior = _parameter_range_prior(
        prior_id="pr_kg_seed", field="leg_length", typical=7.0, confidence=0.6, doi=None
    )
    fresh_prior = _parameter_range_prior(
        prior_id="pr_E1_leg_length", field="leg_length", typical=7.0, confidence=0.52, doi=None
    )

    remaining = _reconcile_kg_literature_duplicates([kg_prior], [fresh_prior])

    assert remaining == [fresh_prior]  # left as a separate prior, not merged
    assert kg_prior.confidence == 0.6  # untouched
    assert kg_prior.notes is None


def test_different_doi_is_left_alone_even_with_matching_value():
    kg_prior = _parameter_range_prior(
        prior_id="pr_kg_seed", field="leg_length", typical=2.75, confidence=0.5, doi="10.1/paper-a"
    )
    fresh_prior = _parameter_range_prior(
        prior_id="pr_E9_leg_length", field="leg_length", typical=2.75, confidence=0.5, doi="10.2/paper-b"
    )

    remaining = _reconcile_kg_literature_duplicates([kg_prior], [fresh_prior])

    assert remaining == [fresh_prior]


def test_same_doi_but_disagreeing_values_is_left_alone_as_a_real_anomaly():
    kg_prior = _parameter_range_prior(
        prior_id="pr_kg_seed", field="leg_length", typical=2.75, confidence=0.5, doi="10.1/real"
    )
    fresh_prior = _parameter_range_prior(
        prior_id="pr_E2_leg_length", field="leg_length", typical=5.0, confidence=0.6, doi="10.1/real"
    )

    remaining = _reconcile_kg_literature_duplicates([kg_prior], [fresh_prior])

    assert remaining == [fresh_prior]
    assert kg_prior.confidence == 0.5  # untouched — not silently resolved either way
