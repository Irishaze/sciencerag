"""One-time (or periodic) batch job: run literature-priors queries against
the internal corpus and queue the results as knowledge-graph candidates for
human approval — the spec's documented "cold-start acceleration (seeding)"
mechanism (docs/spec/sciencerag_spec_zh.md §3.3: "不等仿真攒数据,先跑一个
一次性批处理任务,让 LLM 通读内部论文库,把文献中的关键结论逐条抽取成三元组
...流程完全复用4.4已定义的候选→审批→入库路径,不引入新机制").

Nothing here writes to the graph — it queues candidates the exact same way
POST /sciencerag/validate does (sciencerag/validate/kg_candidate_store.py).
Review and approve with the existing tool, same as any other candidate
batch:

    uv run python scripts/seed_kg_from_corpus.py
    uv run python scripts/approve_kg_candidates.py --list-pending
    uv run python scripts/approve_kg_candidates.py --pending <stem> --list
    uv run python scripts/approve_kg_candidates.py --pending <stem> --approve-all

Real retrieval + LLM calls, one per query (same cost/latency as one
POST /sciencerag/priors call — seconds to ~2 minutes each), so the default
query list is deliberately short. Pass --queries to cover more of the
corpus; run it again later (e.g. after adding new papers) to pick up more
literature — already-queued/approved candidates merge or conflict through
the same add_triple() logic every other write path uses, nothing here
special-cases re-runs.
"""

from __future__ import annotations

import argparse

from sciencerag.priors.retrieval import build_priors_response
from sciencerag.validate import kg_candidate_store
from sciencerag.validate.literature_seeding import prior_to_kg_candidates

DEFAULT_QUERIES = [
    "Bi2Te3 thermoelectric cooler leg length effect on cooling performance",
    "Bi2Te3 thermoelectric cooler pitch and thermal resistance",
    "Bi2Te3 material Seebeck coefficient and figure of merit",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--queries", nargs="+", default=None, help="override the default seed query list"
    )
    parser.add_argument("--max-priors", type=int, default=5, help="max priors per query (default 5)")
    args = parser.parse_args()
    queries = args.queries or DEFAULT_QUERIES

    all_candidates = []
    skipped_kinds: dict[str, int] = {}
    failed_queries: list[str] = []
    for query in queries:
        print(f"querying: {query!r}")
        # Each iteration is a real network + LLM call (seconds to ~2 minutes,
        # per this script's own docstring) — confirmed via a real
        # adversarial test that an unhandled failure on query N (e.g. an LLM
        # API timeout) crashed the whole script and discarded every
        # candidate already extracted from queries 1..N-1, forcing a full
        # expensive re-run to get back what had already been earned. Isolate
        # each query so one network fault costs only that query's work.
        try:
            response, _filtered = build_priors_response(query, allow_external=False, max_priors=args.max_priors)
        except Exception as e:  # noqa: BLE001 - a real retrieval/LLM call, any failure mode is possible
            print(f"  -> FAILED: {e!r} (skipping this query, keeping candidates already collected)")
            failed_queries.append(query)
            continue
        for prior in response.priors:
            candidates = prior_to_kg_candidates(prior)
            if not candidates:
                skipped_kinds[prior.kind] = skipped_kinds.get(prior.kind, 0) + 1
                continue
            all_candidates.extend(candidates)
        print(f"  -> {len(response.priors)} prior(s) extracted, gaps={response.coverage.gaps}")

    if failed_queries:
        print(f"\n{len(failed_queries)} quer{'y' if len(failed_queries) == 1 else 'ies'} failed and were skipped: {failed_queries}")
        print("re-run with --queries to retry just those")

    if skipped_kinds:
        print(
            f"\nskipped (kind not yet mapped to a graph candidate shape — "
            f"see literature_seeding.py's docstring): {skipped_kinds}"
        )

    if not all_candidates:
        print("\nno candidates produced — nothing queued")
        return

    path = kg_candidate_store.store_pending_candidates("lit_seed_batch", all_candidates)
    print(f"\nqueued {len(all_candidates)} candidate(s) -> {path}")
    print("review with: uv run python scripts/approve_kg_candidates.py --list-pending")


if __name__ == "__main__":
    main()
