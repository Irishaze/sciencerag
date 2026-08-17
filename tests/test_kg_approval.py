"""Unit tests for sciencerag.validate.kg_approval's pure source/evidence
attribution logic — source_for() and _evidence_detail_for().

Regression coverage for a real gap found 2026-08-17: a literature-derived
KGCandidate with no traceable DOI (source_for() falls back to
KGSource(type="run", run_id=...), an opaque internal id) used to end up
with evidence_detail=None too — the original Prior's `notes` (paper
title/description, per extract.py's fallback) was sitting right there in
supporting_evidence and got silently discarded on approval, leaving the
resulting KGTriple with zero recoverable provenance.
"""

from sciencerag.validate.kg_approval import _evidence_detail_for, source_for
from sciencerag.validate.models import KGCandidate


def _candidate(**overrides) -> KGCandidate:
    base = dict(
        subject="Bi2Te3 single-stage TEC",
        relation="literature_range_leg_length",
        object_value=7.0,
        object_unit="mm",
        conditions={},
        confidence=0.6,
        run_id="lit_seed_pr_E1_leg_length",
        dedup_status="new",
        entity_type="TECDesign",
        supporting_evidence={},
    )
    base.update(overrides)
    return KGCandidate.model_validate(base)


def test_source_for_uses_doi_when_present():
    candidate = _candidate(supporting_evidence={"prior_id": "pr_E1_leg_length", "source_doi": "10.1/real"})
    source = source_for(candidate)
    assert source.type == "paper"
    assert source.doi == "10.1/real"


def test_source_for_falls_back_to_run_id_when_doi_missing():
    candidate = _candidate(supporting_evidence={"prior_id": "pr_E1_leg_length"})
    source = source_for(candidate)
    assert source.type == "run"
    assert source.run_id == "lit_seed_pr_E1_leg_length"


def test_evidence_detail_passes_through_deviation_detail_for_simulation_candidates():
    candidate = _candidate(supporting_evidence={"deviation_detail": {"verdict": "consistent", "relative_deviation": 0.01}})
    assert _evidence_detail_for(candidate) == {"verdict": "consistent", "relative_deviation": 0.01}


def test_evidence_detail_is_none_when_doi_already_gives_real_traceability():
    # A real DOI is already recoverable provenance — nothing extra to carry.
    candidate = _candidate(
        supporting_evidence={"source_doi": "10.1/real", "notes": "Some Paper Title", "prior_id": "pr_E2_leg_length"}
    )
    assert _evidence_detail_for(candidate) is None


def test_evidence_detail_preserves_notes_and_prior_id_when_doi_is_missing():
    # The real gap this fix closes: no DOI AND no fallback evidence_detail
    # used to mean the resulting KGTriple was completely untraceable.
    candidate = _candidate(
        supporting_evidence={
            "prior_id": "pr_E1_leg_length",
            "prior_kind": "parameter_range",
            "notes": "Reported ideal Bi2Te3 leg length for high cooling performance near 0.7 A.",
        }
    )
    detail = _evidence_detail_for(candidate)
    assert detail == {
        "prior_id": "pr_E1_leg_length",
        "prior_kind": "parameter_range",
        "notes": "Reported ideal Bi2Te3 leg length for high cooling performance near 0.7 A.",
    }


def test_evidence_detail_is_none_when_nothing_at_all_to_carry():
    candidate = _candidate(supporting_evidence={})
    assert _evidence_detail_for(candidate) is None
