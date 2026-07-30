"""M1 threshold-recalibration Step 5a: independent LLM judge.

Reads data/threshold_collection.json (Step 4's 100-query pre-filter dump)
and has GPT-4o (independent of the DeepSeek that scores relevance/drives
extraction in the live pipeline) label two kinds of samples:

  - "boundary": the actual decision zone of the current thresholds
    (evidence relevance in [0.5, 0.7], prior confidence in [0.3, 0.5]).
  - "extrapolation_check": samples OUTSIDE the boundary, where the pipeline
    currently just ASSUMES "high score = good, low score = bad" without
    ever having verified it. If this assumption doesn't hold (KEEP rate
    for relevance>0.7, or DROP rate for relevance<0.5, is well under
    ~90%), Step 6a/5d's plan to build on that assumption is unsound and
    needs to be revisited before proceeding.

Real GPT-4o API calls, but each is a single short completion (not an
agentic PaperQA2 trajectory) — cheap and fast, not hours-long.

Output is a FIXED filename (data/judge_results.json), not date-stamped —
collect_threshold_data.py got bitten hard by a date-stamped output path
silently starting a fresh empty file across a day boundary, breaking
resume logic. A fixed random seed makes the sampling itself deterministic
across re-runs so resuming (skipping already-judged sample ids) stays
correctly aligned with the original sample selection.

    uv run python scripts/threshold_judge.py          # trial: judges 3 samples
    uv run python scripts/threshold_judge.py --full    # full run
"""

import json
import random
import sys
from pathlib import Path

import litellm

COLLECTION_PATH = Path("data/threshold_collection.json")
OUT_PATH = Path("data/judge_results.json")

JUDGE_MODEL = "gpt-4o"
RANDOM_SEED = 42

EVIDENCE_BOUNDARY_CAP = 40
EVIDENCE_EXTRAP_CAP = 25
PRIOR_BOUNDARY_CAP = 40
PRIOR_EXTRAP_CAP = 20

EVIDENCE_RUBRIC = """You are a strict quality judge for a scientific retrieval system about
thermoelectric coolers (TEC).

Given a research query and one evidence snippet, decide whether the snippet
contains SUBSTANTIVE technical information that helps answer the query.

Judge KEEP only if the snippet states concrete technical content: mechanisms,
numeric values, parameter effects, design findings, or material data relevant
to the query.

Judge DROP if the snippet is any of:
- bibliographic/administrative content (reference lists, acknowledgments,
  nomenclature, author info)
- speculation inferred only from titles of cited references, not from the
  paper's own body ("Reference 31 examines... implying...")
- generic statements with no technical content ("COP is important for TEC design")
- content about a clearly different topic

Output JSON only: {"verdict": "KEEP" | "DROP", "reason": "<one sentence>"}"""

PRIOR_RUBRIC = """You are a strict quality judge for structured scientific priors extracted
from TEC literature.

Given: a research query, one extracted prior (kind/field/value/notes), and
the full text of the evidence snippets it cites.

Judge KEEP only if BOTH hold:
1. SUPPORTED: every claim and every number in the prior appears in or follows
   directly from the cited evidence text (no outside knowledge, no invented
   direction/recommendation)
2. USEFUL: the prior carries actionable information for TEC design
   (a value, range, direction, configuration, or concrete caveat — not an
   empty restatement like "X influences performance")

Output JSON only:
{"verdict": "KEEP" | "DROP",
 "failed_check": null | "SUPPORTED" | "USEFUL",
 "reason": "<one sentence>"}"""


def _strip_code_fences(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _build_pools(collection: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flatten the per-query collection into (evidence_pool, prior_pool),
    each item carrying a stable `_id` for resume tracking and the fields
    the judge prompts need."""
    evidence_pool = []
    prior_pool = []

    for entry in collection:
        query = entry["query"]
        evidence_by_key = {}
        for ev in entry["all_evidence"]:
            key = (ev.get("doi") or "", ev.get("span") or "")
            evidence_by_key[key] = ev
            evidence_pool.append(
                {
                    "_id": f"ev|{query}|{key[0]}|{key[1]}",
                    "query": query,
                    "text": ev["text"],
                    "doi": ev.get("doi"),
                    "span": ev.get("span"),
                    "relevance": ev["relevance"],
                }
            )

        for prior in entry["all_priors"]:
            cited_texts = []
            for src in prior.get("sources", []):
                key = (src.get("doi") or "", src.get("span") or "")
                match = evidence_by_key.get(key)
                if match:
                    cited_texts.append(match["text"])
            prior_pool.append(
                {
                    "_id": f"pr|{prior['prior_id']}|{query}",
                    "query": query,
                    "prior_id": prior["prior_id"],
                    "kind": prior["kind"],
                    # sim-contract sync (spec §3.6): scaling_relationship/
                    # candidate_config priors carry their parameter names in
                    # related_fields with field=null — read both, or the
                    # judge output shows a bare "field: null" for exactly
                    # those two kinds and loses which parameters this prior
                    # is actually about.
                    "field": prior["field"],
                    "related_fields": prior.get("related_fields", []),
                    "value": prior["value"],
                    "notes": prior.get("notes"),
                    "confidence": prior["confidence"],
                    "evidence_text": "\n\n".join(cited_texts),
                }
            )

    return evidence_pool, prior_pool


def _sample(pool: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if len(pool) <= cap:
        return list(pool)
    return rng.sample(pool, cap)


def _build_samples(evidence_pool: list[dict], prior_pool: list[dict]) -> list[dict]:
    rng = random.Random(RANDOM_SEED)

    ev_boundary = [e for e in evidence_pool if 0.5 <= e["relevance"] <= 0.7]
    ev_high = [e for e in evidence_pool if e["relevance"] > 0.7]
    ev_low = [e for e in evidence_pool if e["relevance"] < 0.5]

    pr_boundary = [p for p in prior_pool if 0.3 <= p["confidence"] <= 0.5]
    pr_low = [p for p in prior_pool if p["confidence"] < 0.3]
    pr_mid = [p for p in prior_pool if 0.5 <= p["confidence"] < 0.7]
    pr_high = [p for p in prior_pool if p["confidence"] >= 0.7]

    samples = []
    for item in _sample(ev_boundary, EVIDENCE_BOUNDARY_CAP, rng):
        samples.append({**item, "side": "evidence", "sample_type": "boundary", "zone": "0.5-0.7"})
    for item in _sample(ev_high, EVIDENCE_EXTRAP_CAP, rng):
        samples.append(
            {**item, "side": "evidence", "sample_type": "extrapolation_check", "zone": ">0.7"}
        )
    for item in _sample(ev_low, EVIDENCE_EXTRAP_CAP, rng):
        samples.append(
            {**item, "side": "evidence", "sample_type": "extrapolation_check", "zone": "<0.5"}
        )
    for item in _sample(pr_boundary, PRIOR_BOUNDARY_CAP, rng):
        samples.append({**item, "side": "prior", "sample_type": "boundary", "zone": "0.3-0.5"})
    for item in _sample(pr_low, PRIOR_EXTRAP_CAP, rng):
        samples.append(
            {**item, "side": "prior", "sample_type": "extrapolation_check", "zone": "0-0.3"}
        )
    for item in _sample(pr_mid, PRIOR_EXTRAP_CAP, rng):
        samples.append(
            {**item, "side": "prior", "sample_type": "extrapolation_check", "zone": "0.5-0.7"}
        )
    for item in _sample(pr_high, PRIOR_EXTRAP_CAP, rng):
        samples.append(
            {**item, "side": "prior", "sample_type": "extrapolation_check", "zone": "0.7-1.0"}
        )

    return samples


def _judge_one(sample: dict) -> dict:
    if sample["side"] == "evidence":
        system_prompt = EVIDENCE_RUBRIC
        user_prompt = f"Query: {sample['query']}\n\nEvidence snippet:\n{sample['text']}"
    else:
        system_prompt = PRIOR_RUBRIC
        prior_desc = (
            f"kind: {sample['kind']}\nfield: {sample['field']}\n"
            f"related_fields: {sample.get('related_fields', [])}\n"
            f"value: {json.dumps(sample['value'])}\nnotes: {sample['notes']}"
        )
        user_prompt = (
            f"Query: {sample['query']}\n\nExtracted prior:\n{prior_desc}\n\n"
            f"Cited evidence text:\n{sample['evidence_text']}"
        )

    response = litellm.completion(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content
    parsed = json.loads(_strip_code_fences(raw))

    entry = {
        "_id": sample["_id"],
        "side": sample["side"],
        "sample_type": sample["sample_type"],
        "zone": sample["zone"],
        "query": sample["query"],
        "score": sample["relevance"] if sample["side"] == "evidence" else sample["confidence"],
        "verdict": parsed["verdict"],
        "reason": parsed.get("reason"),
        "raw_output": raw,
    }
    if sample["side"] == "evidence":
        entry["text"] = sample["text"]
        entry["doi"] = sample["doi"]
    else:
        entry["prior_id"] = sample["prior_id"]
        entry["kind"] = sample["kind"]
        entry["field"] = sample["field"]
        entry["related_fields"] = sample.get("related_fields", [])
        entry["failed_check"] = parsed.get("failed_check")
    return entry


def main() -> None:
    full_run = "--full" in sys.argv

    collection = json.loads(COLLECTION_PATH.read_text())
    evidence_pool, prior_pool = _build_pools(collection)
    print(f"evidence pool: {len(evidence_pool)} | prior pool: {len(prior_pool)}")

    samples = _build_samples(evidence_pool, prior_pool)
    print(f"total samples selected: {len(samples)}")

    if not full_run:
        samples = samples[:3]
        print(f"TRIAL RUN: judging only {len(samples)} samples (pass --full for the real run)")

    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    done_ids = {r["_id"] for r in results}

    for i, sample in enumerate(samples, 1):
        if sample["_id"] in done_ids:
            continue
        print(f"[{i}/{len(samples)}] ({sample['side']}/{sample['sample_type']}/{sample['zone']}) "
              f"{sample['query'][:60]}")
        try:
            entry = _judge_one(sample)
            print(f"  -> {entry['verdict']} ({entry['reason']})")
        except Exception as e:  # noqa: BLE001 - keep going past a single bad sample
            print(f"  -> FAILED: {type(e).__name__}: {e}")
            continue
        results.append(entry)
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nDone. {len(results)} judged results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
