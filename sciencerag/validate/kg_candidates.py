"""4.4 knowledge-graph candidate extraction (spec §4.4).

Only extracts from runs that passed evaluation. "deviation_found" is
deliberately excluded: spec §4.2 treats a prior/benchmark deviation as "本次
运行最有信息量的发现,需要在输出中显式标出...交给人来裁断" — a finding for a
human to adjudicate, not yet a confirmed fact to propose into the graph. A
"insufficient_benchmark" verdict IS still eligible (unconfirmed but not
contradicted either) with correspondingly lower confidence, since with
today's ~30-sample benchmark set requiring "consistent" for every candidate
would make 4.4 almost always empty in practice.
"""

from __future__ import annotations

from sciencerag.priors.kg import DEFAULT_ENTITY_TYPE, compute_entity_id, query_kg
from sciencerag.priors.ontology_generator import classify_relation, describe_relation, load_ontology
from sciencerag.validate import tec_bridge
from sciencerag.validate.models import Evaluation, KGCandidate, ValidateRequest

# insufficient_benchmark has no per-field deviation to grade against (no
# matching benchmark case at all — see evaluation.py's _benchmark_deviations)
# so it stays a flat constant; a 2026-08-14 sweep of real geometry-distance
# data from logs/audit.jsonl for this verdict came back unusable (test/demo
# payloads with degenerate/extreme parameter values, one causing a division
# blow-up to an astronomical number) — not a real basis to grade by
# "how close was the nearest case", so this is left honestly un-graduated
# rather than fit to noise.
_INSUFFICIENT_BENCHMARK_CONFIDENCE = 0.4

# "consistent" DOES have real per-field deviation data (reference_min/
# reference_max, hence relative_deviation) to grade against — see
# _consistent_confidence below. This min/max range brackets it: 0.5 is
# deliberately just above _INSUFFICIENT_BENCHMARK_CONFIDENCE (a barely-
# consistent match should still outrank an unconfirmed one, but only
# barely), 0.9 is the ceiling for an exact match.
_CONSISTENT_CONFIDENCE_MIN = 0.5
_CONSISTENT_CONFIDENCE_MAX = 0.9


def _matching_benchmark_deviation(evaluation: Evaluation, field: str):
    for deviation in evaluation.deviations:
        if deviation.source == "benchmark_comparison" and deviation.field == field:
            return deviation
    return None


def _deviation_detail(evaluation: Evaluation, field: str) -> dict | None:
    """The concrete numbers behind a field's confidence (2026-08-14/15,
    "方案B": don't invent a new confidence formula that hides the real
    evidence behind one more number — carry the actual relative deviation
    and which benchmark case it was checked against alongside confidence,
    so a downstream reader (a human, or /ask's answer-synthesis LLM) can
    see how strong a given confidence value's support actually is instead
    of trusting the number blind. None when there's no benchmark_comparison
    deviation for this field (insufficient_benchmark, or a defensive gap —
    see _consistent_confidence's own fallback)."""
    deviation = _matching_benchmark_deviation(evaluation, field)
    if deviation is None or deviation.reference_min is None or deviation.reference_max is None:
        return None
    reference = (deviation.reference_min + deviation.reference_max) / 2
    if abs(reference) < 1e-12:
        return None
    return {
        "verdict": evaluation.verdict,
        "relative_deviation": round(abs(deviation.actual - reference) / abs(reference), 4),
        "benchmark_case_id": deviation.reference_id,
    }


def _consistent_confidence(evaluation: Evaluation, field: str) -> float:
    """Real per-field deviation, not a flat 0.7 (2026-08-14/15 finding,
    scripts/loo_scalar_error_sweep.py): two runs both verdicted "consistent"
    can be a near-exact match or barely inside the tolerance band
    (evaluation.py's BENCHMARK_SCALAR_RELATIVE_TOLERANCE_BY_FIELD) — treating
    them identically threw away real information already computed by the
    benchmark comparison. Scales linearly from _CONSISTENT_CONFIDENCE_MAX at
    zero deviation down to _CONSISTENT_CONFIDENCE_MIN at the tolerance edge;
    reference_min/reference_max already encode whichever per-field tolerance
    was actually used, so this doesn't need to duplicate that table.

    Falls back to the old flat constant if this field has no matching
    benchmark_comparison Deviation — shouldn't normally happen when the
    overall verdict is "consistent" (every scalar_results field failing to
    find one would itself have prevented that verdict), but a defensive
    fallback costs nothing and avoids a KeyError-shaped surprise if the
    verdict/deviations ever disagree."""
    deviation = _matching_benchmark_deviation(evaluation, field)
    if deviation is not None and deviation.reference_min is not None and deviation.reference_max is not None:
        reference = (deviation.reference_min + deviation.reference_max) / 2
        tolerance_band = (deviation.reference_max - deviation.reference_min) / 2
        if abs(reference) >= 1e-12 and tolerance_band >= 1e-12:
            relative_deviation = abs(deviation.actual - reference) / abs(reference)
            relative_tolerance = tolerance_band / abs(reference)
            fraction_of_tolerance_used = min(1.0, relative_deviation / relative_tolerance)
            span = _CONSISTENT_CONFIDENCE_MAX - _CONSISTENT_CONFIDENCE_MIN
            return round(_CONSISTENT_CONFIDENCE_MAX - fraction_of_tolerance_used * span, 2)
    return 0.7  # pre-2026-08-14 flat value, only reached if no matching deviation exists

# spec §3.6: material is fixed for this project (Bi2Te3, prior_target=false).
SUBJECT = "Bi2Te3 single-stage TEC"

# A separate entity from SUBJECT, deliberately: SUBJECT is "a TEC design",
# MATERIAL_SUBJECT is "the material it's made of". Both are fixed constants
# today (spec §3.6: material isn't varied), but keeping them distinct
# entities from day one is what lets multiple designs actually connect in
# the rendered graph — every design's SimulationRun links to this SAME
# Material entity (compute_entity_id("Bi2Te3", {}) never changes), instead
# of each design only ever having private edges to its own value nodes
# (the "disconnected star clusters" complaint this fixes).
MATERIAL_SUBJECT = "Bi2Te3"
_MATERIAL_ENTITY_TYPE = "Material"
_SIMULATION_RUN_ENTITY_TYPE = "SimulationRun"

# Hardcoded rather than routed through describe_relation(): these two
# relation names are our own code's structural vocabulary, not something
# the simulator reports and could vary — unlike the achieves_* relations,
# there's no actual ambiguity for AI to resolve here.
_USES_DESIGN_DESCRIPTION = "评估设计"
_USES_MATERIAL_DESCRIPTION = "所用材料"

# Above this KGHit.relevance, treat an existing graph hit as the same claim.
# query_kg is still a stub returning zero hits (sciencerag/priors/kg.py) —
# this threshold can't be exercised against a real duplicate yet, only
# against the empty-hits cold-start path; see kg_candidates.py's own note
# in models.KGCandidate.dedup_status.
_DUPLICATE_RELEVANCE_THRESHOLD = 0.8


def extract_kg_candidates(
    request: ValidateRequest, evaluation: Evaluation
) -> list[KGCandidate]:
    if evaluation.verdict not in ("consistent", "insufficient_benchmark"):
        return []
    if not request.scalar_results:
        return []

    conditions = dict(request.design_parameters)
    conditions["n_pairs"] = float(request.n_pairs)
    supporting_evidence = {
        "scalar_results": request.scalar_results,
        "design_parameters": request.design_parameters,
        "evaluation_verdict": evaluation.verdict,
    }

    # AI-generated ontology (scripts/generate_kg_ontology.py) is a
    # deliberate one-time setup step, not something a hot request path
    # should ever block on or fail over — if nobody's run it yet, every
    # candidate just gets DEFAULT_ENTITY_TYPE / no description, same
    # fallback add_triple() itself uses when neither is supplied.
    ontology = load_ontology()

    candidates = []
    for field, value in request.scalar_results.items():
        relation = f"achieves_{field}"
        hits = query_kg(f"{SUBJECT} {relation}")
        dedup_status = "new"
        if any(hit.relevance >= _DUPLICATE_RELEVANCE_THRESHOLD for hit in hits):
            dedup_status = "duplicate_confirmed"
        entity_type = classify_relation(relation, ontology) if ontology else DEFAULT_ENTITY_TYPE
        relation_description = describe_relation(relation, ontology) if ontology else None
        confidence = (
            _consistent_confidence(evaluation, field)
            if evaluation.verdict == "consistent"
            else _INSUFFICIENT_BENCHMARK_CONFIDENCE
        )
        deviation_detail = _deviation_detail(evaluation, field)
        field_supporting_evidence = (
            {**supporting_evidence, "deviation_detail": deviation_detail}
            if deviation_detail is not None
            else supporting_evidence
        )
        candidates.append(
            KGCandidate(
                subject=SUBJECT,
                relation=relation,
                object_value=value,
                object_unit=tec_bridge.SCALAR_UNITS.get(field),
                conditions=conditions,
                confidence=confidence,
                run_id=request.run_id,
                dedup_status=dedup_status,
                supporting_evidence=field_supporting_evidence,
                entity_type=entity_type,
                relation_description=relation_description,
            )
        )

    # Link candidates: a SimulationRun entity (one per run_id — always a
    # fresh, never-before-seen entity_id, since run_id is unique) pointing
    # at the TECDesign it evaluated and the Material it used. These are
    # structural facts, not measured values, so no dedup_status/query_kg
    # check the way the numeric candidates above get one — the value that
    # matters here is that every run using the same design or the same
    # material resolves to the SAME target entity_id (kg.compute_entity_id
    # is deterministic), which is what actually connects multiple designs
    # in the rendered graph instead of each one only ever having private
    # edges to its own value nodes.
    run_subject = f"Simulation run {request.run_id}"
    design_entity_id = compute_entity_id(SUBJECT, conditions)
    material_entity_id = compute_entity_id(MATERIAL_SUBJECT, {})
    candidates.append(
        KGCandidate(
            subject=run_subject,
            relation="SIMULATION_USES_DESIGN",
            object_entity_id=design_entity_id,
            object_entity_label=SUBJECT,
            object_entity_type="TECDesign",
            relation_description=_USES_DESIGN_DESCRIPTION,
            conditions={},
            confidence=1.0,
            run_id=request.run_id,
            dedup_status="new",
            supporting_evidence=supporting_evidence,
            entity_type=_SIMULATION_RUN_ENTITY_TYPE,
        )
    )
    candidates.append(
        KGCandidate(
            subject=run_subject,
            relation="SIMULATION_USES_MATERIAL",
            object_entity_id=material_entity_id,
            object_entity_label=MATERIAL_SUBJECT,
            object_entity_type=_MATERIAL_ENTITY_TYPE,
            relation_description=_USES_MATERIAL_DESCRIPTION,
            conditions={},
            confidence=1.0,
            run_id=request.run_id,
            dedup_status="new",
            supporting_evidence=supporting_evidence,
            entity_type=_SIMULATION_RUN_ENTITY_TYPE,
        )
    )
    return candidates
