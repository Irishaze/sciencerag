"""One-time data collection for the MIN_EVIDENCE_RELEVANCE / CONFIDENCE_THRESHOLD
recalibration effort (see the "补充任务" section of the M1 plan).

Runs the same 100 real queries as regression_probe_100.py, but — unlike that
script, which only keeps the post-filter survivors — this one persists the
FULL pre-filter picture via PipelineTrace:

  - trace.all_evidence: every PaperQA2 context converted to an EvidenceItem
    (text/doi/span/relevance), including the ones MIN_EVIDENCE_RELEVANCE
    would normally drop silently.
  - trace.all_priors: every LLM-extracted Prior (kind/field/value/notes/
    confidence/sources) BEFORE CONFIDENCE_THRESHOLD splits them into
    strong/weak.
  - trace.confidence_breakdown: the confidence math (base * avg_relevance)
    behind each of those priors.

This is real API calls (DeepSeek extraction + OpenAI embeddings), costs
money, and takes a similar amount of time to regression_probe_100.py
(~1.5-2.5 hours for all 100). Supports resuming: rerun the same command and
already-completed queries are skipped.

    uv run python scripts/collect_threshold_data.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regression_probe_100 import QUERIES  # noqa: E402 - needs sys.path tweak above

from sciencerag.priors.retrieval import build_priors_response_with_trace  # noqa: E402

# Fixed filename, NOT date-stamped: this is one evolving dataset collected
# (and re-collected, on failures) over however many real sessions it takes
# — a date-derived name meant the moment the resume run crossed a calendar
# day boundary, `date.today()` silently pointed at a brand-new empty file,
# resume/skip logic saw zero done_queries, and the whole 100-query batch
# got redone from scratch (found the hard way — burned a real rerun on it).
DEFAULT_OUT_PATH = Path("data/threshold_collection.json")


def main(
    queries: list[tuple[str, str]] | None = None, out_path: Path = DEFAULT_OUT_PATH
) -> None:
    if queries is None:
        queries = QUERIES
        assert len(queries) == 100, f"expected 100 queries, got {len(queries)}"

    out_path.parent.mkdir(exist_ok=True)

    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
    done_queries = {r["query"] for r in results}

    for i, (category, query) in enumerate(queries, 1):
        if query in done_queries:
            continue
        print(f"\n[{i}/{len(queries)}] ({category}) {query}")
        t0 = time.time()
        try:
            response, trace = build_priors_response_with_trace(query)
            elapsed = time.time() - t0
            entry = {
                "category": category,
                "query": query,
                "elapsed_s": round(elapsed, 1),
                "internal_hits": response.coverage.internal_hits,
                "n_survived_priors": len(response.priors),
                "all_evidence": [item.model_dump() for item in trace.all_evidence],
                "all_priors": [p.model_dump() for p in trace.all_priors],
                "confidence_breakdown": [
                    cb.model_dump() for cb in trace.confidence_breakdown
                ],
                "gaps": response.coverage.gaps,
                "error": None,
            }
            print(
                f"  -> {len(trace.all_evidence)} evidence "
                f"({len(response.priors)} of {len(trace.all_priors)} priors survived), "
                f"{elapsed:.0f}s"
            )
        except Exception as e:  # noqa: BLE001 - capture and continue past any single failure
            elapsed = time.time() - t0
            entry = {
                "category": category,
                "query": query,
                "elapsed_s": round(elapsed, 1),
                "internal_hits": None,
                "n_survived_priors": None,
                "all_evidence": [],
                "all_priors": [],
                "confidence_breakdown": [],
                "gaps": [],
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"  -> HARD FAILURE: {type(e).__name__}: {e}")

        results.append(entry)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n\nDone. {len(results)} results saved to {out_path}")
    n_errors = sum(1 for r in results if r["error"])
    total_evidence = sum(len(r["all_evidence"]) for r in results)
    total_priors = sum(len(r["all_priors"]) for r in results)
    print(
        f"hard failures: {n_errors}  |  total evidence contexts: {total_evidence}  |  "
        f"total extracted priors (pre-filter): {total_priors}"
    )


if __name__ == "__main__":
    main()
