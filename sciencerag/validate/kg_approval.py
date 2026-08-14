"""Shared KG-candidate approval logic (spec §6.3: "候选 -> 审批 -> 入库" is
the only path into the graph) — used by both scripts/approve_kg_candidates.py
(CLI, spec §7's original v1 substitute for a page) and
sciencerag/kg_approval/router.py (the web panel). One implementation so the
CLI and the API can never drift on what "approving a candidate" actually
does — same source attribution, same audit logging, same add_triple() call.
"""

from __future__ import annotations

from typing import Literal

from sciencerag.common.audit import log_audit_entry
from sciencerag.common.trace import new_trace_id
from sciencerag.priors.kg import KGSource, KGTriple, add_triple
from sciencerag.validate.models import KGCandidate


def source_for(candidate: KGCandidate) -> KGSource:
    """Literature-derived candidates (sciencerag/validate/
    literature_seeding.py) stash the originating paper's DOI in
    supporting_evidence["source_doi"] rather than adding a new KGCandidate
    field — attribute those triples to the paper, not to a simulation run
    that never happened."""
    doi = candidate.supporting_evidence.get("source_doi")
    if doi:
        return KGSource(type="paper", doi=doi)
    return KGSource(type="run", run_id=candidate.run_id)


def approve_candidate(
    candidate: KGCandidate, operator: str, reason: str
) -> tuple[KGTriple, Literal["added", "merged", "conflict"]]:
    """The only function that ever calls add_triple() for a KGCandidate —
    raises ValueError straight through (e.g. add_triple's non-finite-value
    gate) so callers decide for themselves whether one bad candidate should
    abort a batch or just get skipped."""
    triple, status = add_triple(
        subject=candidate.subject,
        relation=candidate.relation,
        object_value=candidate.object_value,
        object_unit=candidate.object_unit,
        object_entity_id=candidate.object_entity_id,
        object_entity_label=candidate.object_entity_label,
        object_entity_type=candidate.object_entity_type,
        relation_description=candidate.relation_description,
        evidence_detail=candidate.supporting_evidence.get("deviation_detail"),
        conditions=candidate.conditions,
        confidence=candidate.confidence,
        run_id=candidate.run_id,
        sources=[source_for(candidate)],
        entity_type=candidate.entity_type,
    )
    log_audit_entry(
        trace_id=new_trace_id("kgappr"),
        endpoint="sciencerag.kg_approval",
        request={"candidate": candidate.model_dump(), "operator": operator, "reason": reason},
        evidence=[],
        output={"triple": triple.model_dump(), "status": status},
    )
    return triple, status
