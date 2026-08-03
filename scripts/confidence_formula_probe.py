"""k/relevance/confidence plan Part 2 step 3: offline (zero API cost)
comparison of 3 candidate confidence formulas against the v3 judged prior
data (see k-relevance-abstract-lark.md).

Reads data/threshold_collection_v3_post_numeric_gate.json (Part 2 step 1's
48-query pre-filter dump, collected after numeric_check Phase B3 went
live) and data/judge_results_v3.json (step 2's independent GPT-4o KEEP/DROP
judgments on the prior side of that data), recomputes three candidate
scores per judged prior, and buckets KEEP rate by score for each — same
bucketing approach as threshold_curve.py, not reusing its code directly
since that script hardcodes a single score field ("confidence") where this
one needs three.

Candidate A — noisy-OR over distinct sources:
    score_A = 1 - prod_{d in distinct sources}(1 - r_d)
    r_d = max relevance among a source's own cited snippets. "Distinct
    source" mirrors extract.py's own n_papers convention: same DOI merges,
    a missing DOI never merges with another missing DOI (each counted
    separately, since span/DOI is all we have to disambiguate them here —
    same simplification extract.py already makes for n_papers).

Candidate B — A x consistency:
    consistency defaults to 1.0 with <2 distinct sources, or when the
    prior asserts zero numbers (scaling_relationship/caution priors never
    do — see numeric_check.extract_numbers), or when no pair of distinct
    sources both independently corroborate at least one of the prior's own
    numbers (nothing pairable to compare). Otherwise: for every pair of
    distinct sources where BOTH sources' own evidence text contains a
    number matching (numbers_match, 2% tolerance) at least one of the
    prior's claimed numbers, the pair "agrees" if their matched-number sets
    overlap, "conflicts" if not. consistency = agreeing_pairs /
    comparable_pairs, floored at 0.5 (per the plan: a disagreement is a
    downweight signal, not grounds to zero the prior out — SUPPORTED/
    fabrication is numeric_check's job, not this formula's).
    This is a real, but rough, reconstruction of the plan's "consistency"
    description — flagged here for review, not asserted as ground truth.

Candidate C — control:
    the CURRENT production formula, unrefit. Requires zero recomputation:
    prior["confidence"] in the collection IS exactly what that formula
    produced when this data was collected. A genuine refit of its 5
    constants (as originally planned) is deliberately NOT attempted here —
    with only ~20 judged prior samples (and, going in, only a handful of
    DROPs), fitting 5 free parameters would be noise-fitting, not
    recalibration; reported plainly as a finding, not silently skipped.

    uv run python scripts/confidence_formula_probe.py
"""

import json
import math
from pathlib import Path

from sciencerag.priors.models import Prior
from sciencerag.priors.numeric_check import (
    extract_numbers,
    extract_numbers_from_text,
    numbers_match,
)

COLLECTION_PATH = Path("data/threshold_collection_v3_post_numeric_gate.json")
JUDGE_PATH = Path("data/judge_results_v3.json")

CONSISTENCY_FLOOR = 0.5


def _source_identity(doi: str | None, index: int) -> str:
    return doi if doi else f"_nodoi_{index}"


def _items_for_prior(entry: dict, prior: dict) -> list[dict]:
    """Reconstruct which all_evidence items back this prior, by matching
    (doi, span) against prior['sources'] — the same key scheme
    threshold_judge.py's _build_pools uses to join sources back to text."""
    evidence_by_key = {}
    for ev in entry["all_evidence"]:
        key = (ev.get("doi") or "", ev.get("span") or "")
        evidence_by_key[key] = ev
    items = []
    for src in prior.get("sources", []):
        key = (src.get("doi") or "", src.get("span") or "")
        match = evidence_by_key.get(key)
        if match:
            items.append(match)
    return items


def _group_by_source(items: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for i, item in enumerate(items):
        key = _source_identity(item.get("doi"), i)
        grouped.setdefault(key, []).append(item)
    return grouped


def score_a(items: list[dict]) -> float:
    grouped = _group_by_source(items)
    prod = 1.0
    for group in grouped.values():
        r_d = max(item["relevance"] for item in group)
        prod *= 1 - r_d
    return round(1 - prod, 4)


def _consistency(items: list[dict], prior_obj: Prior) -> float:
    grouped = _group_by_source(items)
    if len(grouped) < 2:
        return 1.0
    prior_numbers = extract_numbers(prior_obj)
    if not prior_numbers:
        return 1.0

    matched_by_source: dict[str, set[float]] = {}
    for key, group in grouped.items():
        text = " ".join(item["text"] for item in group)
        source_numbers = extract_numbers_from_text(text)
        matched = {n for n in prior_numbers if any(numbers_match(n, sn) for sn in source_numbers)}
        if matched:
            matched_by_source[key] = matched

    keys = sorted(matched_by_source)
    comparable = 0
    agreeing = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            comparable += 1
            a, b = matched_by_source[keys[i]], matched_by_source[keys[j]]
            if any(numbers_match(x, y) for x in a for y in b):
                agreeing += 1

    if comparable == 0:
        return 1.0
    return max(CONSISTENCY_FLOOR, round(agreeing / comparable, 4))


def score_b(items: list[dict], prior_obj: Prior) -> float:
    return round(score_a(items) * _consistency(items, prior_obj), 4)


def _bucket(score: float, width: float) -> float:
    idx = math.floor(round(score / width, 6))
    return round(idx * width, 2)


def _print_table(rows: list[tuple], headers: list[str]) -> None:
    widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths, strict=True)))


def _report(name: str, scored: list[tuple[float, str]]) -> None:
    print()
    print("=" * 70)
    print(f"Candidate {name}: KEEP rate by 0.05-wide score bucket (n={len(scored)})")
    print("=" * 70)
    buckets: dict[float, list[str]] = {}
    for score, verdict in scored:
        buckets.setdefault(_bucket(score, 0.05), []).append(verdict)
    rows = []
    for b in sorted(buckets):
        verdicts = buckets[b]
        n = len(verdicts)
        keep_rate = sum(1 for v in verdicts if v == "KEEP") / n
        rows.append((f"[{b:.2f}, {b + 0.05:.2f})", n, f"{keep_rate:.1%}"))
    _print_table(rows, ["bucket", "n", "KEEP rate"])

    print()
    print(f"-- {name} cumulative (score >= X) --")
    rows = []
    for x in sorted({round(s, 2) for s, _ in scored}):
        subset = [v for s, v in scored if s >= x]
        keep_rate = sum(1 for v in subset if v == "KEEP") / len(subset)
        rows.append((f">= {x}", len(subset), f"{keep_rate:.1%}"))
    _print_table(rows, ["score", "n", "KEEP rate"])


def main() -> None:
    collection = json.loads(COLLECTION_PATH.read_text())
    judged = json.loads(JUDGE_PATH.read_text())

    prior_by_key = {}
    for entry in collection:
        for prior in entry["all_priors"]:
            prior_by_key[(prior["prior_id"], entry["query"])] = (entry, prior)

    prior_judgments = [r for r in judged if r["side"] == "prior"]
    print(f"{len(prior_judgments)} judged prior samples")

    scores_a, scores_b, scores_c = [], [], []
    unmatched = 0
    for r in prior_judgments:
        key = (r["prior_id"], r["query"])
        if key not in prior_by_key:
            unmatched += 1
            continue
        entry, prior = prior_by_key[key]
        items = _items_for_prior(entry, prior)
        if not items:
            unmatched += 1
            continue
        prior_obj = Prior.model_validate(prior)

        verdict = r["verdict"]
        scores_a.append((score_a(items), verdict))
        scores_b.append((score_b(items, prior_obj), verdict))
        scores_c.append((prior["confidence"], verdict))  # current production formula, as-collected

    if unmatched:
        print(f"WARNING: {unmatched} judged samples could not be matched back to source data")

    _report("A (noisy-OR)", scores_a)
    _report("B (noisy-OR x consistency)", scores_b)
    _report("C (current additive formula, UNREFIT constants — see module docstring)", scores_c)


if __name__ == "__main__":
    main()
