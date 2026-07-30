"""M1 threshold-recalibration Step 5b: misclassification-rate curve.

Reads data/judge_results.json (Step 5a's full judge output — BOTH
sample_type="boundary" and "extrapolation_check", pooled together so the
curve spans the full score range, not just the old 0.5-0.7 decision zone)
and buckets KEEP rate by score, for evidence (relevance) and priors
(confidence) separately. Prior buckets are further split by failed_check
(SUPPORTED vs USEFUL failures) among the DROPs.

Evidence relevance is only ever a multiple of 0.1 (PaperQA2's underlying
score is an integer 1-10) — bucketing that at 0.05 width would just
alternate half-empty buckets, so evidence uses 0.1-wide buckets. Prior
confidence is a continuous formula output, so it uses the planned 0.05
width.

    uv run python scripts/threshold_curve.py
"""

import json
import math
from pathlib import Path

JUDGE_PATH = Path("data/judge_results.json")


def _bucket(score: float, width: float) -> float:
    # Plain `score // width` misfiles exact bucket boundaries: 0.5 // 0.1
    # is 4.0, not 5.0, because 0.5/0.1 rounds to 4.999999999999999 in
    # float64 — silently put boundary scores in the bucket BELOW where
    # they belong. Round the quotient first to kill that noise, then floor.
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


def main() -> None:
    data = json.loads(JUDGE_PATH.read_text())

    print("=" * 70)
    print("EVIDENCE side (relevance -> KEEP rate), 0.1-wide buckets")
    print("all sample_type pooled: boundary (0.5-0.7) + extrapolation_check (<0.5, >0.7)")
    print("=" * 70)
    evidence = [r for r in data if r["side"] == "evidence"]
    buckets: dict[float, list[str]] = {}
    for r in evidence:
        b = _bucket(r["score"], 0.1)
        buckets.setdefault(b, []).append(r["verdict"])
    rows = []
    for b in sorted(buckets):
        verdicts = buckets[b]
        n = len(verdicts)
        keep_rate = sum(1 for v in verdicts if v == "KEEP") / n
        rows.append((f"[{b:.1f}, {b+0.1:.1f})", n, f"{keep_rate:.1%}"))
    _print_table(rows, ["bucket", "n", "KEEP rate"])

    print()
    print("=" * 70)
    print("PRIOR side (confidence -> KEEP rate), 0.05-wide buckets")
    print("all sample_type pooled: boundary (0.3-0.5) + extrapolation_check (0.5-0.7, 0.7-1.0)")
    print("=" * 70)
    priors = [r for r in data if r["side"] == "prior"]
    buckets = {}
    for r in priors:
        b = _bucket(r["score"], 0.05)
        buckets.setdefault(b, []).append(r)
    rows = []
    for b in sorted(buckets):
        items = buckets[b]
        n = len(items)
        keep_rate = sum(1 for r in items if r["verdict"] == "KEEP") / n
        drops = [r for r in items if r["verdict"] == "DROP"]
        n_supported_fail = sum(1 for r in drops if r.get("failed_check") == "SUPPORTED")
        n_useful_fail = sum(1 for r in drops if r.get("failed_check") == "USEFUL")
        rows.append(
            (
                f"[{b:.2f}, {b+0.05:.2f})",
                n,
                f"{keep_rate:.1%}",
                n_supported_fail,
                n_useful_fail,
            )
        )
    _print_table(rows, ["bucket", "n", "KEEP rate", "DROP:SUPPORTED-fail", "DROP:USEFUL-fail"])

    print()
    print("=" * 70)
    print("Cumulative view: KEEP rate for score >= X (helps spot a threshold, not just a bucket)")
    print("=" * 70)
    print("-- evidence --")
    rows = []
    for x in sorted({round(r["score"], 2) for r in evidence}):
        subset = [r for r in evidence if r["score"] >= x]
        keep_rate = sum(1 for r in subset if r["verdict"] == "KEEP") / len(subset)
        rows.append((f">= {x}", len(subset), f"{keep_rate:.1%}"))
    _print_table(rows, ["score", "n", "KEEP rate"])

    print()
    print("-- prior --")
    rows = []
    for x in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        subset = [r for r in priors if r["score"] >= x]
        if not subset:
            continue
        keep_rate = sum(1 for r in subset if r["verdict"] == "KEEP") / len(subset)
        rows.append((f">= {x}", len(subset), f"{keep_rate:.1%}"))
    _print_table(rows, ["score", "n", "KEEP rate"])


if __name__ == "__main__":
    main()
