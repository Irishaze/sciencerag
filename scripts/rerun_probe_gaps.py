"""Rerun just the queries whose results were compromised by the ~2-day
unattended run of regression_probe_100.py (laptop sleep/network interrupts
causing spurious zero-hit results). Updates
scripts/_regression_probe_100_results.json in place, leaving the other
(good) entries untouched.

    uv run python scripts/rerun_probe_gaps.py
"""

import json
import time
from pathlib import Path

from sciencerag.priors.retrieval import build_priors_response_with_trace

RESULTS_PATH = Path("scripts/_regression_probe_100_results.json")

# Identified as suspect: internal_hits==0 (retrieval found nothing — the
# main symptom of network interruption), evidence wiped by the 0.7 filter,
# or a hard extraction error.
QUERIES_TO_RERUN = [
    "What are the drawbacks of using thermoelectric generators for waste heat recovery?",
    "What limits the maximum achievable COP in thermoelectric cooling?",
    "What are the reliability concerns for thermoelectric coolers in aerospace applications?",
    "What are the tradeoffs of increasing driving current in thermoelectric coolers?",
    "What are the limitations of segmented thermoelectric leg designs?",
    "How are thermoelectric coolers used in PCR thermal cyclers for nucleic acid amplification?",
    "How are thermoelectric coolers used in wearable body cooling devices?",
    "How are thermoelectric coolers used in battery thermal management systems?",
    "How are thermoelectric coolers used to control laser crystal temperature?",
    "How are thermoelectric generators used for energy harvesting in IoT devices?",
    "How are thermoelectric coolers used in infrared sensor cooling?",
    "How are thermoelectric coolers used in electric motor thermal management?",
    "How are thermoelectric coolers used in atmospheric water harvesting systems?",
    "How are thermoelectric devices used in concrete-based energy harvesting?",
    "How are thermoelectric coolers applied in gravimeter temperature control systems?",
    "How are thermoelectric coolers used in vacuum sensor test systems?",
    "How are thermoelectric generation-cooling systems integrated for waste heat recovery?",
    "How are thermoelectric coolers used for protective clothing thermal comfort?",
    "How are thermoelectric coolers used in agricultural machinery heat utilization?",
    "How are thermoelectric coolers used for LED thermal management?",
    "How does segmented leg design compare to uniform leg design in thermoelectric coolers?",
    "How does two-stage cooling compare to single-stage cooling for large temperature differences?",
    "How does pulse-frequency modulation driving compare to constant DC driving for COP?",
    "How do water-cooled and air-cooled hot side designs compare for thermoelectric coolers?",
    "How does active thermal management compare to passive cooling for electric motors?",
    "How does radiative cooling compare to thermoelectric cooling for daytime applications?",
    "How do pyramid-style and cuboid-style two-stage thermoelectric coolers compare?",
    "How does forced convection cooling compare to thermoelectric cooling for VLSI circuits?",
    "What design parameters affect the coefficient of performance of a thermoelectric cooler?",
    "What factors should be optimized when designing a thermoelectric cooler for battery thermal management?",
    "What are the key considerations for designing a two-stage thermoelectric cooler?",
    "What are the main sources of performance loss in thermoelectric cooling systems?",
    "What thermal management strategies are used across different thermoelectric cooler applications?",
    "What role does control strategy play in thermoelectric cooler performance?",
    "What role do contact interfaces play in thermoelectric device thermal resistance?",
    "What uncertainty and sensitivity analysis methods have been applied to thermoelectric refrigerators?",
    "What is the typical thermal conductance of a thermoelectric cooler heat sink interface?",
    "What sample throughput has been achieved with thermoelectric-cooler-based PCR devices?",
    "What power consumption reduction has been achieved through thermoelectric cooler optimization?",
    "What lifetime or reliability data exists for thermoelectric coolers in continuous operation?",
    "What electrical resistivity values are reported for thermoelectric cooler materials?",
    "How is the figure of merit ZT related to thermoelectric cooler design optimization?",
]


def main() -> None:
    assert len(QUERIES_TO_RERUN) == 42, f"expected 42, got {len(QUERIES_TO_RERUN)}"

    results = json.loads(RESULTS_PATH.read_text())
    by_query = {r["query"]: r for r in results}

    for i, query in enumerate(QUERIES_TO_RERUN, 1):
        print(f"\n[{i}/42] {query}")
        old = by_query.get(query, {})
        category = old.get("category", "unknown")
        t0 = time.time()
        try:
            response, trace = build_priors_response_with_trace(query)
            elapsed = time.time() - t0
            retries = max(len(trace.attempts) - 1, 0)
            entry = {
                "category": category,
                "query": query,
                "elapsed_s": round(elapsed, 1),
                "internal_hits": response.coverage.internal_hits,
                "retries": retries,
                "priors": [
                    {
                        "kind": p.kind,
                        "field": p.field,
                        "confidence": p.confidence,
                        "n_sources": len(p.sources),
                        "dois": [s.doi for s in p.sources],
                        "value": p.value,
                    }
                    for p in response.priors
                ],
                "gaps": response.coverage.gaps,
                "error": None,
                "rerun": True,
            }
            print(f"  -> {len(response.priors)} priors, internal_hits={response.coverage.internal_hits}, {elapsed:.0f}s")
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            entry = {
                "category": category,
                "query": query,
                "elapsed_s": round(elapsed, 1),
                "internal_hits": None,
                "retries": None,
                "priors": [],
                "gaps": [],
                "error": f"{type(e).__name__}: {e}",
                "rerun": True,
            }
            print(f"  -> HARD FAILURE: {type(e).__name__}: {e}")

        by_query[query] = entry
        RESULTS_PATH.write_text(json.dumps(list(by_query.values()), indent=2, ensure_ascii=False))

    print("\n\nDone.")
    still_zero = [r for r in by_query.values() if r["error"] is None and len(r["priors"]) == 0]
    still_errors = [r for r in by_query.values() if r["error"]]
    print(f"still zero-prior after rerun: {len(still_zero)}")
    print(f"still hard errors after rerun: {len(still_errors)}")
    for r in still_zero:
        print("  zero:", r["query"])
    for r in still_errors:
        print("  error:", r["query"], "->", r["error"])


if __name__ == "__main__":
    main()
