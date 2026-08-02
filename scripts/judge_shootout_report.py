"""Phase C Step C3: comparison report over data/judge_shootout_results.json.

Reads scripts/judge_shootout.py's output (no API calls here — pure offline
analysis) and reports, per judge (x format, where applicable):
  - BAD detection rate (/10): correctly verdicted NOT_SUPPORTED
  - GOOD false-rejection rate (/10): incorrectly verdicted NOT_SUPPORTED
  - detection broken down by corruption_type
  - fielded vs natural format sensitivity (API judges only; local_nli is
    natural-only by design, see judge_shootout.py)

Read order per the task spec: check gpt5_6_sol_ceiling's own misses first
(a BAD item even the ceiling model can't catch is a candidate for "the exam
item itself is ambiguous, not a hard case" and should be manually
reviewed/reworded rather than held against the cheaper applicants), THEN
compare the 3 applicants against the ceiling to quantify what going cheap
costs.

    uv run python scripts/judge_shootout_report.py
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("data/judge_shootout_results.json")
CEILING_JUDGE = "gpt5_6_sol_ceiling"
APPLICANT_JUDGES = ["deepseek_self_check", "gpt5_6_luna", "local_nli"]


def _print_table(rows: list[tuple], headers: list[str]) -> None:
    widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths, strict=True)))


def _expected_verdict(true_label: str) -> str:
    return "SUPPORTED" if true_label == "GOOD" else "NOT_SUPPORTED"


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())

    by_judge_fmt: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_judge_fmt[(r["judge"], r["format"])].append(r)

    print("=" * 78)
    print("STEP 1: gpt5_6_sol_ceiling misses (review these exam items first)")
    print("=" * 78)
    ceiling_misses = []
    for fmt in ["fielded", "natural"]:
        for r in by_judge_fmt.get((CEILING_JUDGE, fmt), []):
            if r["verdict"] != _expected_verdict(r["true_label"]):
                ceiling_misses.append(r)
    if not ceiling_misses:
        print("None — the ceiling model got every item right in both formats.")
    else:
        for r in ceiling_misses:
            print(
                f"  {r['item_id']} ({r['format']}): true={r['true_label']}"
                f" ({r['corruption_type']}) -> ceiling said {r['verdict']}"
            )
            print(f"    reason: {r.get('reason')}")
    print()

    print("=" * 78)
    print("STEP 2: BAD detection / GOOD false-rejection, per judge x format")
    print("=" * 78)
    judges_present = sorted({r["judge"] for r in rows})
    table_rows = []
    for judge in judges_present:
        for fmt in ["fielded", "natural"]:
            items = by_judge_fmt.get((judge, fmt))
            if not items:
                continue
            bad = [r for r in items if r["true_label"] == "BAD"]
            good = [r for r in items if r["true_label"] == "GOOD"]
            bad_caught = sum(1 for r in bad if r["verdict"] == "NOT_SUPPORTED")
            good_rejected = sum(1 for r in good if r["verdict"] == "NOT_SUPPORTED")
            table_rows.append(
                (
                    judge,
                    fmt,
                    f"{bad_caught}/{len(bad)}",
                    f"{good_rejected}/{len(good)}",
                )
            )
    _print_table(table_rows, ["judge", "format", "BAD caught", "GOOD false-rejected"])
    print()

    print("=" * 78)
    print("STEP 3: BAD detection by corruption_type, per judge (natural format)")
    print("=" * 78)
    corruption_types = sorted({r["corruption_type"] for r in rows if r["corruption_type"]})
    table_rows = []
    for judge in judges_present:
        items = by_judge_fmt.get((judge, "natural"), [])
        by_type = defaultdict(list)
        for r in items:
            if r["true_label"] == "BAD":
                by_type[r["corruption_type"]].append(r)
        row = [judge]
        for ct in corruption_types:
            group = by_type.get(ct, [])
            caught = sum(1 for r in group if r["verdict"] == "NOT_SUPPORTED")
            row.append(f"{caught}/{len(group)}" if group else "-")
        table_rows.append(tuple(row))
    _print_table(table_rows, ["judge", *corruption_types])
    print()

    print("=" * 78)
    print("STEP 4: applicants vs ceiling — the cost of going cheap")
    print("=" * 78)
    ceiling_bad_caught = {}
    ceiling_good_rejected = {}
    for fmt in ["fielded", "natural"]:
        items = by_judge_fmt.get((CEILING_JUDGE, fmt), [])
        bad = [r for r in items if r["true_label"] == "BAD"]
        good = [r for r in items if r["true_label"] == "GOOD"]
        ceiling_bad_caught[fmt] = sum(1 for r in bad if r["verdict"] == "NOT_SUPPORTED")
        ceiling_good_rejected[fmt] = sum(1 for r in good if r["verdict"] == "NOT_SUPPORTED")

    for judge in APPLICANT_JUDGES:
        for fmt in ["fielded", "natural"]:
            items = by_judge_fmt.get((judge, fmt))
            if not items:
                continue
            bad = [r for r in items if r["true_label"] == "BAD"]
            good = [r for r in items if r["true_label"] == "GOOD"]
            bad_caught = sum(1 for r in bad if r["verdict"] == "NOT_SUPPORTED")
            good_rejected = sum(1 for r in good if r["verdict"] == "NOT_SUPPORTED")
            gap_bad = ceiling_bad_caught.get(fmt, 0) - bad_caught
            gap_good = good_rejected - ceiling_good_rejected.get(fmt, 0)
            print(
                f"  {judge:20s} ({fmt:8s}): BAD caught {bad_caught}/{len(bad)}"
                f" (ceiling {ceiling_bad_caught.get(fmt, '?')}/10, gap={gap_bad:+d}),"
                f" GOOD false-rejected {good_rejected}/{len(good)}"
                f" (ceiling {ceiling_good_rejected.get(fmt, '?')}/10, gap={gap_good:+d})"
            )

    print()
    print("Report complete. This script does NOT pick a judge — that decision")
    print("is the user's, based on the tables above.")


if __name__ == "__main__":
    main()
