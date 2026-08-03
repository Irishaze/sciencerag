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

from sciencerag.priors.kg import query_kg
from sciencerag.validate.models import Evaluation, KGCandidate, ValidateRequest

_UNIT_BY_SCALAR = {
    "delta_T_max_K": "K",
    "optimal_current_A": "A",
    "optimal_voltage_V": "V",
    "total_resistance_ohm": "ohm",
    "max_heat_dissipation_W": "W",
    "figure_of_merit_1_per_K": "1/K",
}

_CONFIDENCE_BY_VERDICT = {
    "consistent": 0.7,  # confirmed against a matching known benchmark case
    "insufficient_benchmark": 0.4,  # unconfirmed but not contradicted
}

# spec §3.6: material is fixed for this project (Bi2Te3, prior_target=false).
SUBJECT = "Bi2Te3 single-stage TEC"

# Above this KGHit.relevance, treat an existing graph hit as the same claim.
# query_kg is still a stub returning zero hits (sciencerag/priors/kg.py) —
# this threshold can't be exercised against a real duplicate yet, only
# against the empty-hits cold-start path; see kg_candidates.py's own note
# in models.KGCandidate.dedup_status.
_DUPLICATE_RELEVANCE_THRESHOLD = 0.8


def extract_kg_candidates(
    request: ValidateRequest, evaluation: Evaluation
) -> list[KGCandidate]:
    if evaluation.verdict not in _CONFIDENCE_BY_VERDICT:
        return []
    if not request.scalar_results:
        return []

    confidence = _CONFIDENCE_BY_VERDICT[evaluation.verdict]
    conditions = dict(request.design_parameters)
    conditions["n_pairs"] = float(request.n_pairs)
    supporting_evidence = {
        "scalar_results": request.scalar_results,
        "design_parameters": request.design_parameters,
        "evaluation_verdict": evaluation.verdict,
    }

    candidates = []
    for field, value in request.scalar_results.items():
        relation = f"achieves_{field}"
        hits = query_kg(f"{SUBJECT} {relation}")
        dedup_status = "new"
        if any(hit.relevance >= _DUPLICATE_RELEVANCE_THRESHOLD for hit in hits):
            dedup_status = "duplicate_confirmed"
        candidates.append(
            KGCandidate(
                subject=SUBJECT,
                relation=relation,
                object_value=value,
                object_unit=_UNIT_BY_SCALAR.get(field),
                conditions=conditions,
                confidence=confidence,
                run_id=request.run_id,
                dedup_status=dedup_status,
                supporting_evidence=supporting_evidence,
            )
        )
    return candidates
