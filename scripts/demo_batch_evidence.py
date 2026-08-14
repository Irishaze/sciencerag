"""Live demo: the full internal process of sciencerag.priors.batch_evidence
for two contrasting candidate designs — literature retrieval, the evidence
table after relevance filtering, the exact prompt sent to the LLM, the raw
LLM response, and the final supports/refutes/neutral classification.

Real API calls throughout (PaperQA2 retrieval + DeepSeek classification),
one retrieval + one LLM call per candidate — small real cost.

    uv run python scripts/demo_batch_evidence.py
"""

from __future__ import annotations

import json

from sciencerag.priors.batch_evidence import (
    SYSTEM_PROMPT,
    CandidateEvidence,
    CandidateSpec,
    EvidenceStance,
    _classify_stances,
)
from sciencerag.priors.retrieval import _build_evidence_table, run_query


def _step(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def run_for_candidate(candidate: CandidateSpec) -> CandidateEvidence:
    _step(f"CANDIDATE {candidate.candidate_id}: {candidate.description!r}")

    print("\n[1] retrieval — batch_evidence re-runs its own PaperQA2 query per")
    print("    candidate (candidate.description as the query string); it does")
    print("    NOT reuse a prior /sciencerag/priors call.")
    response = run_query(candidate.description)
    contexts = response.session.contexts
    print(f"    -> PaperQA2 returned {len(contexts)} raw context(s)")

    evidence_table, below_threshold = _build_evidence_table(contexts)
    print(
        f"\n[2] evidence table after relevance filtering "
        f"(MIN_EVIDENCE_RELEVANCE=0.5): {len(evidence_table)} kept, "
        f"{len(below_threshold)} dropped"
    )
    for label, item in evidence_table.items():
        print(f"    [{label}] doi={item.doi} relevance={item.relevance:.3f}")
        snippet = item.text[:180].replace("\n", " ")
        print(f'        "{snippet}..."')

    if not evidence_table:
        print("\n(no evidence survived filtering -> coverage=insufficient, no LLM call made)")
        return CandidateEvidence(candidate_id=candidate.candidate_id, coverage="insufficient")

    evidence_block = "\n".join(
        f'[{label}] (source: {item.doi or "unknown"}) "{item.text}"'
        for label, item in evidence_table.items()
    )
    user_prompt = (
        f"Candidate configuration: {candidate.description}\n"
        f"Design parameters: {candidate.design_parameters}\n\n"
        f"Evidence:\n{evidence_block}\n\nClassification:"
    )
    print(
        f"\n[3] exact prompt sent to the LLM — ONE call for all "
        f"{len(evidence_table)} evidence item(s) of this candidate (not one call per item):"
    )
    print("    --- system ---")
    print("    " + SYSTEM_PROMPT)
    print("    --- user ---")
    for line in user_prompt.splitlines():
        print("    " + line)

    print("\n[4] calling the LLM to classify stance for each evidence item...")
    stances = _classify_stances(candidate, evidence_table)
    print(f"    raw stance map returned: {json.dumps(stances, indent=2)}")

    print("\n[5] final classification:")
    result = CandidateEvidence(candidate_id=candidate.candidate_id, coverage="ok")
    for label, item in evidence_table.items():
        stance = stances.get(label, "neutral")
        flag = "" if label in stances else "  (defaulted — LLM omitted this label)"
        print(f"    [{label}] -> {stance}{flag}")
        entry = EvidenceStance(
            doi=item.doi, span=item.span, text=item.text, relevance=item.relevance, stance=stance
        )
        if stance == "supports":
            result.supporting.append(entry)
        elif stance == "refutes":
            result.refuting.append(entry)
        else:
            result.neutral.append(entry)
    return result


def main() -> None:
    candidates = [
        CandidateSpec(
            candidate_id="cand_short_legs",
            description="short thermoelectric leg length maximizes COP",
            design_parameters={"leg_length": 0.5},
        ),
        CandidateSpec(
            candidate_id="cand_long_legs",
            description="long thermoelectric leg length maximizes COP",
            design_parameters={"leg_length": 3.0},
        ),
    ]

    results = [run_for_candidate(candidate) for candidate in candidates]

    _step("SUMMARY — what POST /sciencerag/priors/batch_evidence would return")
    for result in results:
        print(
            f"\ncandidate_id={result.candidate_id}  coverage={result.coverage}  "
            f"supporting={len(result.supporting)}  refuting={len(result.refuting)}  "
            f"neutral={len(result.neutral)}"
        )


if __name__ == "__main__":
    main()
