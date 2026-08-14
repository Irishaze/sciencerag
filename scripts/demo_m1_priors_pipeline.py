"""M1 (sciencerag.priors) pipeline walkthrough — every intermediate stage,
not just the final response.

Calls POST /sciencerag/priors/_debug (demo/debug only, not part of the
spec-compliant API — see sciencerag/priors/router.py) to get the real
PipelineTrace alongside the real PriorsResponse, then prints:

  0. the knowledge-graph lookup (spec §3.2: KG first, always additive,
     never a substitute for literature) — which entities matched, and the
     Priors that came out of them. No LLM involved (deterministic keyword
     overlap over the graph, jieba-segmented for CJK query text since
     2026-08-11); PipelineTrace itself never captures this step because
     it happens in retrieval.py before extract_priors() is even called,
     so this script calls the same functions the real request path uses
     (_kg_priors_for_query / query_kg_entities) directly, read-only.
  1. every retrieved evidence snippet with its relevance score
  2. what survives the MIN_EVIDENCE_RELEVANCE=0.5 filter, and what's cut
  3. numeric-groundedness check failures (if any)
  4. the confidence formula's breakdown per surviving prior
  5. the final ranking after the max_priors cap — each entry labeled
     [KG] or [文献] by its prior_id prefix, since the final list is
     kg_priors + literature-extracted priors merged together (retrieval.py's
     _build_priors_response) and that mix is otherwise invisible.

Real API calls throughout (DeepSeek + gpt-5.6-luna + PaperQA2) — same
cost/latency convention as scripts/demo_end_to_end.py.

    uv run python scripts/demo_m1_priors_pipeline.py
    uv run python scripts/demo_m1_priors_pipeline.py "你的问题"
"""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from sciencerag.app import app
from sciencerag.priors.kg import query_kg_entities
from sciencerag.priors.retrieval import _kg_priors_for_query

client = TestClient(app)

QUERY = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Bi2Te3 thermoelectric cooler leg length effect on COP and delta_T_max"
)
MAX_PRIORS = 5


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    print(f"query: {QUERY!r}")
    print(f"max_priors: {MAX_PRIORS}")

    # ------------------------------------------------------------ step 0
    _rule("0  knowledge-graph lookup (no LLM — deterministic keyword overlap)")
    kg_result = query_kg_entities(QUERY, max_entities=MAX_PRIORS)
    if not kg_result.groups:
        print("  no matching entities in the graph for this query")
    else:
        print(
            f"  total_matching_entities={kg_result.total_matching_entities}  "
            f"entities_returned={kg_result.entities_returned}"
        )
        for g in kg_result.groups:
            print(f"  - entity_id={g.entity_id}  relevance={g.relevance:.3f}  ({len(g.triples)} triple(s))")
    kg_priors = _kg_priors_for_query(QUERY)
    if not kg_priors:
        print("\n  -> 0 Prior(s) produced from the graph for this query")
    else:
        print(f"\n  -> {len(kg_priors)} Prior(s) produced from the graph (prior_id starts with 'pr_kg_'):")
        for p in kg_priors:
            print(f"     {p.prior_id}  [{p.kind}]  confidence={p.confidence:.2f}")

    resp = client.post(
        "/sciencerag/priors/_debug",
        json={"query": QUERY, "max_priors": MAX_PRIORS},
    )
    resp.raise_for_status()
    body = resp.json()
    response, trace = body["response"], body["trace"]

    # ---------------------------------------------------------- step 1/5
    _rule("1/5  all retrieved evidence, with relevance (before filtering)")
    all_evidence = trace["all_evidence"]
    survivors = trace["evidence"]
    survivor_keys = {(e["doi"], e["span"], e["text"]) for e in survivors}
    for i, e in enumerate(sorted(all_evidence, key=lambda e: e["relevance"], reverse=True), 1):
        kept = (e["doi"], e["span"], e["text"]) in survivor_keys
        mark = "KEEP" if kept else "cut "
        doi = e["doi"] or "(no doi)"
        print(f"  [{mark}] relevance={e['relevance']:.3f}  {doi}  {e['span']}")
        print(f"        {e['text']}")
    print(f"\n  total retrieved: {len(all_evidence)}")

    # ---------------------------------------------------------- step 2/5
    _rule("2/5  after MIN_EVIDENCE_RELEVANCE=0.5 filter")
    print(f"  survived: {len(survivors)} / {len(all_evidence)}")
    cut = len(all_evidence) - len(survivors)
    if cut:
        print(f"  cut: {cut} snippet(s) below the 0.5 relevance threshold")
    for label, item in zip(trace["evidence_labels"], survivors):
        doi = item["doi"] or "(no doi)"
        print(f"  {label}: relevance={item['relevance']:.3f}  {doi}  {item['span']}")
        print(f"       {item['text']}")

    # ---------------------------------------------------------- step 3/5
    _rule("3/5  numeric-groundedness check")
    failures = trace["numeric_check_failures"]
    if not failures:
        print("  no failures — every number asserted by a surviving prior was")
        print("  found in its cited evidence text")
    else:
        for f in failures:
            print(f"  REJECTED: {f}")
        print(f"\n  {len(failures)} rejection(s) — note: a prior rejected here may still")
        print("  appear below if a later retry attempt produced a grounded version")

    _rule("   raw LLM output per attempt (diagnostic — not part of the 5 steps)")
    for a in trace["attempts"]:
        status = f"ERROR: {a['error']}" if a["error"] else "OK"
        print(f"  --- attempt {a['attempt']} ({status}) ---")
        print(f"  {a['raw_output']}\n")

    # ---------------------------------------------------------- step 4/5
    # confidence is computed INSIDE _to_prior() before the numeric-
    # groundedness check runs (extract.py), so a breakdown entry here is not
    # necessarily a survivor — if the draft's numbers don't check out, it's
    # rejected right after this and never reaches trace["all_priors"].
    # Retries re-run the whole draft set from scratch, so if every attempt
    # ultimately failed (this run's case), NONE of the entries below
    # correspond to a final prior — they're the retry attempts' drafts.
    _rule("4/5  confidence formula breakdown (every attempt's drafts, pre-numeric-check)")
    print("  confidence = round(min(1.0, base * avg_relevance), 2)")
    print("  base = 0.5 + 0.1*min(n_papers,3) + 0.02*min(extra_snippets,3)\n")
    for c in trace["confidence_breakdown"]:
        extra = c["n_snippets"] - c["n_papers"]
        print(
            f"  {c['prior_field'] or '(unnamed)':30s} "
            f"n_papers={c['n_papers']} n_snippets={c['n_snippets']} extra_snippets={extra}  "
            f"base={c['base']:.2f}  avg_relevance={c['avg_relevance']:.2f}  "
            f"-> confidence={c['confidence']:.2f}"
        )
    if not trace["all_priors"]:
        print("\n  none of the drafts above survived — every attempt's numeric or")
        print("  semantic check rejected all of them (see steps 3 and coverage.gaps)")

    # ---------------------------------------------------------- step 5/5
    _rule(f"5/5  final ranking (top {MAX_PRIORS}, sorted by confidence desc)")
    if not response["priors"]:
        print("  (empty — no prior survived every check for this query; see")
        print("   coverage.gaps below for why)")
    for i, p in enumerate(response["priors"], 1):
        origin = "KG" if p["prior_id"].startswith("pr_kg_") else "文献"
        print(f"  #{i}  [{origin}]  confidence={p['confidence']:.2f}  [{p['kind']}] field={p.get('field')}")
        print(f"       value={p['value']}")

    # response["priors"] is kg_priors + literature-extracted priors merged
    # (retrieval.py's _build_priors_response), but trace["all_priors"] only
    # ever captures the literature side (extract_priors' own drafts) — add
    # the KG side back in so this comparison isn't silently off by
    # len(kg_priors).
    all_priors = trace["all_priors"] + kg_priors
    truncated = len(all_priors) - len(response["priors"])
    if truncated > 0:
        print(f"\n  {truncated} additional prior(s) survived all checks but were cut by")
        print(f"  max_priors={MAX_PRIORS} (lower confidence) — recorded in coverage.gaps:")
        for gap in response["coverage"]["gaps"]:
            if "max_priors" in gap:
                print(f"    {gap}")

    _rule("done")
    print(f"coverage: {response['coverage']}")


if __name__ == "__main__":
    main()
