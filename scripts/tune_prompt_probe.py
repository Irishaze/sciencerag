"""M1-13 prompt-tuning probe: run a small diverse batch of real queries and
dump structured, reviewable output (not just a wall of prose) so failure
modes in the extraction prompt are easy to spot by eye.

Not a regression test (no pass/fail assertions) — this is the exploratory
step that precedes M1-17's fixtures. Costs real API calls.

    uv run python scripts/tune_prompt_probe.py
"""

import json

from sciencerag.priors.retrieval import build_priors_response_with_trace

PROBE_QUERIES = [
    ("parameter_range", "What is the optimal driving voltage range for a thermoelectric cooler to maximize COP?"),
    ("material_property", "What are typical Seebeck coefficient and thermal conductivity values for thermoelectric cooler materials?"),
    ("scaling_relationship", "How does COP scale with the temperature difference between hot and cold sides in a thermoelectric cooler?"),
    ("candidate_config", "What driving methods or configurations improve thermoelectric cooler efficiency?"),
    ("caution", "What are the limitations or failure modes of using thermoelectric coolers for battery thermal management?"),
    ("cross-cutting", "What factors limit the maximum achievable COP in thermoelectric cooling?"),
    ("new-corpus-application", "How are thermoelectric coolers used in PCR thermal cyclers for nucleic acid amplification?"),
    ("comparative", "How does segmented leg design compare to uniform leg design in thermoelectric coolers?"),
]


def main() -> None:
    results = []
    for expected_kind, query in PROBE_QUERIES:
        print(f"\n{'=' * 100}\nPROBE [{expected_kind}]: {query}\n{'=' * 100}")
        try:
            response, trace = build_priors_response_with_trace(query)
        except Exception as e:  # noqa: BLE001 - probe script, want to see everything
            print(f"HARD FAILURE: {type(e).__name__}: {e}")
            results.append({"query": query, "expected_kind": expected_kind, "error": str(e)})
            continue

        retries = len(trace.attempts) - 1
        failed_attempts = [a for a in trace.attempts if a.error]

        print(f"internal_hits={response.coverage.internal_hits}  priors_kept={len(response.priors)}  retries={retries}")
        if failed_attempts:
            for a in failed_attempts:
                print(f"  [attempt {a.attempt} FAILED] {a.error}")
        for p in response.priors:
            n_src = len(p.sources)
            print(f"  - [{p.kind:20}] {p.field:28} conf={p.confidence:.2f}  n_sources={n_src}")
            print(f"      value keys: {list(p.value.keys())}")
        if response.coverage.gaps:
            for g in response.coverage.gaps:
                print(f"  gap: {g}")

        results.append(
            {
                "query": query,
                "expected_kind": expected_kind,
                "retries": retries,
                "priors": [
                    {
                        "kind": p.kind,
                        "field": p.field,
                        "confidence": p.confidence,
                        "n_sources": len(p.sources),
                        "value": p.value,
                    }
                    for p in response.priors
                ],
                "gaps": response.coverage.gaps,
            }
        )

    with open("scripts/_probe_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved {len(results)} probe results to scripts/_probe_results.json")


if __name__ == "__main__":
    main()
