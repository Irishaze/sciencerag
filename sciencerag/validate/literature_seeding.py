"""Converts literature Priors (sciencerag/priors/models.py) into knowledge-
graph candidates, for the spec's documented "cold-start acceleration
(seeding)" mechanism (docs/spec/sciencerag_spec_zh.md §3.3: "不等仿真攒数据,
先跑一个一次性批处理任务,让 LLM 通读内部论文库,把文献中的关键结论逐条抽取成
三元组...流程完全复用4.4已定义的候选→审批→入库路径,不引入新机制").

No retrieval, no graph writes — scripts/seed_kg_from_corpus.py drives it
(calling the same priors-retrieval pipeline /sciencerag/priors uses, then
this), and scripts/approve_kg_candidates.py is still the only thing that
ever calls kg.add_triple() — the human-approval gate is untouched by this
feature, same as every other source of KGCandidate.

Does make real (cached) LLM calls now (2026-08-17, describe_relation below)
— this module used to be pure conversion logic, but that left every
literature-seeded triple with no relation_description at all, which is the
ONLY Chinese-language content a triple can carry (kg.py's _render_text
docstring) — a real, reproduced bug: query_kg_entities("腿长") returned 0
hits against a graph that genuinely had a leg_length literature value in
it, while query_kg_entities("leg length") returned 6, purely because
kg_candidates.py (the simulation-result write path) already called
describe_relation() and this path never did.
"""

from __future__ import annotations

import logging

from sciencerag.priors.kg import compute_entity_id
from sciencerag.priors.models import Prior
from sciencerag.priors.ontology_generator import describe_relation, load_ontology
from sciencerag.validate.models import KGCandidate

logger = logging.getLogger(__name__)

# Same fixed subject sciencerag/validate/kg_candidates.py uses for
# simulation-derived candidates — a literature-sourced parameter_range
# candidate describes "what this device type's geometry should generally
# be", not one specific simulated design, so it shares the same subject but
# empty conditions (giving it its own distinct entity_id, never colliding
# with a real simulated design's non-empty conditions).
_SUBJECT = "Bi2Te3 single-stage TEC"

# scaling_relationship priors describe a relationship BETWEEN two design
# parameters (e.g. leg_length vs thermal_resistance) — neither parameter is
# a TECDesign or a Material, and the AI-generated ontology (data/kg/
# ontology.json) doesn't currently define a dedicated "parameter" entity
# type, so this reuses its documented fallback category rather than
# inventing an ad hoc name outside the ontology.
_PARAMETER_ENTITY_TYPE = "GenericParameter"

_DIRECTION_RELATION = {
    "positive": "POSITIVELY_CORRELATES_WITH",
    "negative": "NEGATIVELY_CORRELATES_WITH",
    "convex": "CORRELATES_WITH",
    "concave": "CORRELATES_WITH",
    "non_monotonic": "CORRELATES_WITH",
    "unknown": "CORRELATES_WITH",
}
_DIRECTION_DESCRIPTION_ZH = {
    "positive": "正相关",
    "negative": "负相关",
    "convex": "非线性相关",
    "concave": "非线性相关",
    "non_monotonic": "非单调相关",
    "unknown": "存在关联",
}


def _first_doi(prior: Prior) -> str | None:
    for source in prior.sources:
        doi = getattr(source, "doi", None)
        if doi:
            return doi
    return None


def _numeric_conditions(conditions: dict[str, str | float]) -> dict[str, float]:
    """ParameterRangeValue/MaterialPropertyValue.conditions is dict[str, str
    | float] (extract.py lets the LLM note qualitative context too, e.g.
    material_state="annealed") but KGCandidate.conditions is dict[str,
    float] only (it feeds compute_entity_id, which needs to hash cleanly).
    Real gap this closes (2026-08-17): this used to be discarded entirely
    (every literature_range_* candidate got conditions={}), which collapsed
    every literature fact about the same field onto ONE entity_id
    regardless of what circumstance it was reported under — e.g. "leg
    length at 0.7A applied current" and "leg length, no stated condition"
    ended up hashed as the same claim, so kg.py's add_triple flagged them
    as agreeing/conflicting with each other when they were actually
    talking about two different things that were never comparable to begin
    with. Keeping the real numeric conditions gives them distinct entity_ids
    instead."""
    return {k: v for k, v in conditions.items() if isinstance(v, (int, float))}


def _relation_description_for(relation: str) -> str | None:
    """Best-effort, matching kg_candidates.py's own defensive pattern for
    the exact same real LLM call: a network hiccup here must degrade to no
    description, never crash the whole seeding batch — this only ever runs
    from scripts/seed_kg_from_corpus.py, not a request-serving hot path,
    but the failure mode (and the cache load_ontology()/describe_relation()
    already do) is identical."""
    ontology = load_ontology()
    if not ontology:
        return None
    try:
        return describe_relation(relation, ontology)
    except Exception as e:  # noqa: BLE001 - see docstring
        logger.warning("describe_relation failed for %r, leaving relation_description unset: %s", relation, e)
        return None


def _supporting_evidence(prior: Prior) -> dict:
    """scripts/approve_kg_candidates.py reads `source_doi` out of this to
    attribute the resulting triple to the paper it came from
    (KGSource(type="paper", doi=...)) instead of the type="run" provenance
    simulation-derived candidates get — no schema change needed since
    KGCandidate.supporting_evidence is already a free-form field for
    exactly this kind of "支撑数据切片"."""
    evidence: dict = {"prior_id": prior.prior_id, "prior_kind": prior.kind}
    doi = _first_doi(prior)
    if doi:
        evidence["source_doi"] = doi
    if prior.notes:
        evidence["notes"] = prior.notes
    return evidence


def prior_to_kg_candidates(prior: Prior) -> list[KGCandidate]:
    """Best-effort conversion — not every Prior kind maps cleanly onto a
    (subject, relation, object) triple:

    - parameter_range / material_property: a straightforward numeric fact.
    - scaling_relationship: a link between two parameter entities, direction
      encoded in the relation name (this is the actual answer to "为什么A和B
      的关系进不了图谱" — it's a link candidate, not a numeric one).
    - candidate_config (a whole bundle of parameters reported together) and
      caution (a free-text warning, not a fact) don't fit either shape
      without losing what actually made them useful — skipped for now
      rather than forced. Callers should log what got skipped (see
      scripts/seed_kg_from_corpus.py), not silently drop it.
    """
    run_id = f"lit_seed_{prior.prior_id}"
    evidence = _supporting_evidence(prior)

    if prior.kind == "parameter_range":
        v = prior.value
        value = v.typical
        if value is None and v.min is not None and v.max is not None:
            value = (v.min + v.max) / 2
        if value is None:
            value = v.min if v.min is not None else v.max
        if value is None:
            return []
        relation = f"literature_range_{v.field_name}"
        return [
            KGCandidate(
                subject=_SUBJECT,
                relation=relation,
                object_value=value,
                object_unit=v.unit,
                conditions=_numeric_conditions(v.conditions),
                confidence=prior.confidence,
                run_id=run_id,
                dedup_status="new",
                supporting_evidence=evidence,
                relation_description=_relation_description_for(relation),
                entity_type="TECDesign",
            )
        ]

    if prior.kind == "material_property":
        v = prior.value
        if v.magnitude is None:
            return []
        relation = f"literature_{v.property_name}"
        return [
            KGCandidate(
                subject=v.material,
                relation=relation,
                object_value=v.magnitude,
                object_unit=v.unit,
                conditions=_numeric_conditions(v.conditions),
                confidence=prior.confidence,
                run_id=run_id,
                dedup_status="new",
                supporting_evidence=evidence,
                relation_description=_relation_description_for(relation),
                entity_type="Material",
            )
        ]

    if prior.kind == "scaling_relationship":
        v = prior.value
        return [
            KGCandidate(
                subject=v.x,
                relation=_DIRECTION_RELATION[v.direction],
                object_entity_id=compute_entity_id(v.y, {}),
                object_entity_label=v.y,
                object_entity_type=_PARAMETER_ENTITY_TYPE,
                relation_description=_DIRECTION_DESCRIPTION_ZH[v.direction],
                conditions={},
                confidence=prior.confidence,
                run_id=run_id,
                dedup_status="new",
                supporting_evidence=evidence,
                entity_type=_PARAMETER_ENTITY_TYPE,
            )
        ]

    return []  # candidate_config, caution — not yet mapped, see docstring
