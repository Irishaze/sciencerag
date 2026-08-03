"""k/relevance/confidence plan Part 2 — candidate D: LLM-scored confidence.

User's call (2026-08-04, after A/B/C showed no separation on v3 data —
see confidence_formula_probe.py): have an independent LLM assign a
continuous 0-1 confidence score directly, instead of computing one from
n_papers/relevance. Explicitly against the principle recorded in
extract.py's module docstring ("Confidence is not LLM-scored... per spec's
'don't fake a calibrated probability' principle") — kept as a real,
data-backed comparison rather than either silently blocking it or silently
adopting it. The resulting scores get bucketed against the SAME v3 judge
verdicts A/B/C were, via confidence_formula_probe.py's --llm flag, so this
is an apples-to-apples 4th candidate, not a separate unvalidated claim.

Uses gpt-5.6-luna specifically (Phase C's judge shootout winner, see
data/judge_shootout_results.json / the phase_c_judge_shootout_decision
memory), not SCIENCERAG_LLM_MODEL — scoring a prior with the SAME model
that extracted it would be the model grading its own homework, exactly
what the "not LLM-scored" principle was originally trying to avoid; an
independent model at least avoids that specific failure mode.

Real API calls, but single short completions per prior (not agentic
PaperQA2 trajectories) — cheap and fast for the ~22-prior v3 set.
Resumable: rerun and already-scored (prior_id, query) pairs are skipped.

    uv run python scripts/llm_confidence_score.py
"""

import json
from pathlib import Path

import litellm

from sciencerag.priors.extract import _strip_code_fences

COLLECTION_PATH = Path("data/threshold_collection_v3_post_numeric_gate.json")
JUDGE_PATH = Path("data/judge_results_v3.json")
OUT_PATH = Path("data/llm_confidence_scores_v3.json")

MODEL = "gpt-5.6-luna"

RUBRIC = """You are scoring how well-CORROBORATED a scientific claim (a "prior") is by
its cited evidence, for a thermoelectric cooler (TEC) research assistant.

You will be given a research query, an extracted prior (kind/field/value/
notes), and the full text of every evidence snippet cited as its source.

Score confidence from 0.0 to 1.0 based on how strongly the cited evidence
corroborates this specific claim:
- Higher: precise, unambiguous support; multiple INDEPENDENT sources
  (different papers) stating the same or compatible values; the claim is
  specific (a value, range, or concrete configuration) not vague.
- Lower: support from only a single source; the evidence is indirect,
  approximate, or requires inference; the claim is generic/non-specific;
  multiple sources exist but disagree with each other.

This is about corroboration STRENGTH, not about whether the claim is
literally true — a single-source, precisely-stated, directly-supported
claim can still score moderately high; a claim needing real inferential
leaps from the evidence should score low even if it seems plausible.

Output JSON only, no markdown fences:
{"confidence": <float 0.0-1.0>, "reason": "<one sentence>"}"""


def _score_one(query: str, prior: dict, evidence_text: str) -> dict:
    prior_desc = (
        f"kind: {prior['kind']}\nfield: {prior['field']}\n"
        f"related_fields: {prior.get('related_fields', [])}\n"
        f"value: {json.dumps(prior['value'])}\nnotes: {prior.get('notes')}"
    )
    messages = [
        {"role": "system", "content": RUBRIC},
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\nExtracted prior:\n{prior_desc}\n\n"
                f"Cited evidence text:\n{evidence_text}"
            ),
        },
    ]
    try:
        response = litellm.completion(model=MODEL, messages=messages, temperature=0)
    except litellm.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        response = litellm.completion(model=MODEL, messages=messages)
    raw = response.choices[0].message.content
    return json.loads(_strip_code_fences(raw))


def main() -> None:
    collection = json.loads(COLLECTION_PATH.read_text())
    judged = json.loads(JUDGE_PATH.read_text())

    prior_by_key = {}
    for entry in collection:
        for prior in entry["all_priors"]:
            prior_by_key[(prior["prior_id"], entry["query"])] = (entry, prior)

    prior_judgments = [r for r in judged if r["side"] == "prior"]

    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    done = {(r["prior_id"], r["query"]) for r in results}

    for i, r in enumerate(prior_judgments, 1):
        key = (r["prior_id"], r["query"])
        if key in done:
            continue
        if key not in prior_by_key:
            continue
        entry, prior = prior_by_key[key]
        evidence_by_key = {
            (ev.get("doi") or "", ev.get("span") or ""): ev for ev in entry["all_evidence"]
        }
        cited_texts = []
        for src in prior.get("sources", []):
            match = evidence_by_key.get((src.get("doi") or "", src.get("span") or ""))
            if match:
                cited_texts.append(match["text"])
        evidence_text = "\n\n".join(cited_texts)

        print(f"[{i}/{len(prior_judgments)}] {prior['prior_id']} ({r['query'][:50]})")
        try:
            parsed = _score_one(r["query"], prior, evidence_text)
            print(f"  -> {parsed['confidence']} ({parsed.get('reason')})")
        except Exception as e:  # noqa: BLE001 - keep going past a single bad sample
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            continue

        results.append(
            {
                "prior_id": prior["prior_id"],
                "query": r["query"],
                "judge_verdict": r["verdict"],
                "llm_confidence": parsed["confidence"],
                "reason": parsed.get("reason"),
            }
        )
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(results)} scored results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
