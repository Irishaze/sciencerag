"""M1-17 raw material: run 100 diverse real queries against the (fixed)
extraction pipeline and save full structured results.

This is NOT the final regression fixture set — it's the evidence base you
curate INTO fixtures (pick representative queries, write property-based
"expect" assertions per the spec §8 guidance: assert kind/DOI/field
presence, not exact text). Costs real API calls; takes ~1.5-2.5 hours.

    uv run python scripts/regression_probe_100.py
"""

import json
import time
from pathlib import Path

from sciencerag.priors.retrieval import build_priors_response_with_trace

QUERIES: list[tuple[str, str]] = [
    # --- parameter_range targeted ---
    ("parameter_range", "What is the optimal driving voltage for a thermoelectric cooler to maximize COP?"),
    ("parameter_range", "What operating current range maximizes cooling capacity in a thermoelectric module?"),
    ("parameter_range", "What is a typical leg length or leg height used in thermoelectric cooler design?"),
    ("parameter_range", "What temperature difference range is typical for thermoelectric coolers in battery thermal management?"),
    ("parameter_range", "What is the optimal ratio of thermoelectric module numbers in a two-stage cooler?"),
    ("parameter_range", "What duty cycle is used in pulse-frequency modulation driving of thermoelectric coolers?"),
    ("parameter_range", "What is the typical cooling capacity (in watts) of thermoelectric coolers used in PCR thermal cyclers?"),
    ("parameter_range", "What ambient temperature range has been tested for thermoelectric cooler extreme temperature control?"),
    ("parameter_range", "What contact resistance values are reported for thermoelectric cooler interfaces?"),
    ("parameter_range", "What is the typical thermal conductance of a thermoelectric cooler heat sink interface?"),
    ("parameter_range", "What input power levels are used for wearable thermoelectric cooling devices?"),
    ("parameter_range", "What cross-sectional area is typical for thermoelectric cooler legs?"),
    # --- material_property targeted ---
    ("material_property", "What is the Seebeck coefficient of thermoelectric materials used in coolers?"),
    ("material_property", "What is the thermal conductivity of Bi2Te3-based thermoelectric materials?"),
    ("material_property", "What electrical resistivity values are reported for thermoelectric cooler materials?"),
    ("material_property", "What is the figure of merit ZT for thermoelectric cooling materials?"),
    ("material_property", "What material properties affect the Thomson effect in thermoelectric devices?"),
    ("material_property", "What are the material properties of Mg3Bi2-based thermoelectric coolers?"),
    ("material_property", "What material properties are used in segmented thermoelectric leg design?"),
    ("material_property", "What temperature-dependent material properties matter for thermoelectric cooler modeling?"),
    ("material_property", "What is the Peltier coefficient of thermoelectric cooler materials?"),
    ("material_property", "What material properties are extracted using non-linear least squares methods for thermoelectric coolers?"),
    # --- scaling_relationship targeted ---
    ("scaling_relationship", "How does COP scale with the temperature difference between hot and cold sides?"),
    ("scaling_relationship", "How does cooling capacity change with driving current in a thermoelectric cooler?"),
    ("scaling_relationship", "How does contact resistance affect the coefficient of performance?"),
    ("scaling_relationship", "How does leg geometry affect thermal resistance and electrical resistance?"),
    ("scaling_relationship", "How does blower fan speed relate to COP in thermoelectric cooling systems?"),
    ("scaling_relationship", "How does the number of thermoelectric stages affect achievable temperature difference?"),
    ("scaling_relationship", "How does ambient temperature affect thermoelectric cooler performance?"),
    ("scaling_relationship", "How does input power relate to cooling performance in wearable thermoelectric coolers?"),
    ("scaling_relationship", "How does segment length affect COP in segmented thermoelectric legs?"),
    ("scaling_relationship", "How does asymmetric contact resistance location affect thermoelectric cooler performance?"),
    # --- candidate_config targeted ---
    ("candidate_config", "What driving methods improve thermoelectric cooler efficiency compared to constant DC?"),
    ("candidate_config", "What heat sink designs are used to improve thermoelectric cooler performance?"),
    ("candidate_config", "What multistage configurations are used for large temperature difference cooling?"),
    ("candidate_config", "What leg geometries have been explored to improve thermoelectric cooler COP?"),
    ("candidate_config", "What control strategies are used for extreme temperature control in thermoelectric coolers?"),
    ("candidate_config", "What designs combine heat pipes with thermoelectric coolers for battery thermal management?"),
    ("candidate_config", "What configurations are used for thermoelectric coolers in laser crystal temperature control?"),
    ("candidate_config", "What hybrid cooling designs combine thermoelectric coolers with phase change materials?"),
    ("candidate_config", "What driving current waveforms improve thermoelectric cooler performance?"),
    ("candidate_config", "What fan integration designs are used with thermoelectric coolers?"),
    # --- caution targeted ---
    ("caution", "What are the limitations of thermoelectric coolers for battery thermal management?"),
    ("caution", "What failure modes affect thermoelectric coolers in portable diagnostic devices?"),
    ("caution", "What are the drawbacks of using thermoelectric generators for waste heat recovery?"),
    ("caution", "What limits the maximum achievable COP in thermoelectric cooling?"),
    ("caution", "What are the reliability concerns for thermoelectric coolers in aerospace applications?"),
    ("caution", "What are the tradeoffs of increasing driving current in thermoelectric coolers?"),
    ("caution", "What are the limitations of segmented thermoelectric leg designs?"),
    # --- application-specific (probing the expanded 135-paper corpus breadth) ---
    ("application", "How are thermoelectric coolers used in PCR thermal cyclers for nucleic acid amplification?"),
    ("application", "How are thermoelectric coolers used in wearable body cooling devices?"),
    ("application", "How are thermoelectric coolers used in battery thermal management systems?"),
    ("application", "How are thermoelectric coolers used to control laser crystal temperature?"),
    ("application", "How are thermoelectric generators used for energy harvesting in IoT devices?"),
    ("application", "How are thermoelectric coolers used in infrared sensor cooling?"),
    ("application", "How are thermoelectric coolers used in electric motor thermal management?"),
    ("application", "How are thermoelectric coolers used in atmospheric water harvesting systems?"),
    ("application", "How are thermoelectric devices used in concrete-based energy harvesting?"),
    ("application", "How are thermoelectric coolers applied in gravimeter temperature control systems?"),
    ("application", "How are thermoelectric coolers used in vacuum sensor test systems?"),
    ("application", "How are thermoelectric generation-cooling systems integrated for waste heat recovery?"),
    ("application", "How are thermoelectric coolers used for protective clothing thermal comfort?"),
    ("application", "How are thermoelectric coolers used in agricultural machinery heat utilization?"),
    ("application", "How are thermoelectric coolers used for LED thermal management?"),
    # --- comparative ---
    ("comparative", "How does segmented leg design compare to uniform leg design in thermoelectric coolers?"),
    ("comparative", "How does two-stage cooling compare to single-stage cooling for large temperature differences?"),
    ("comparative", "How does pulse-frequency modulation driving compare to constant DC driving for COP?"),
    ("comparative", "How do water-cooled and air-cooled hot side designs compare for thermoelectric coolers?"),
    ("comparative", "How does active thermal management compare to passive cooling for electric motors?"),
    ("comparative", "How does radiative cooling compare to thermoelectric cooling for daytime applications?"),
    ("comparative", "How do pyramid-style and cuboid-style two-stage thermoelectric coolers compare?"),
    ("comparative", "How does forced convection cooling compare to thermoelectric cooling for VLSI circuits?"),
    # --- broad / synthesis ---
    ("broad", "What design parameters affect the coefficient of performance of a thermoelectric cooler?"),
    ("broad", "What factors should be optimized when designing a thermoelectric cooler for battery thermal management?"),
    ("broad", "What are the key considerations for designing a two-stage thermoelectric cooler?"),
    ("broad", "What are the main sources of performance loss in thermoelectric cooling systems?"),
    ("broad", "What thermal management strategies are used across different thermoelectric cooler applications?"),
    ("broad", "What role does control strategy play in thermoelectric cooler performance?"),
    ("broad", "What are the engineering tradeoffs in miniaturized thermoelectric cooler design?"),
    ("broad", "What experimental methods are used to characterize thermoelectric cooler performance?"),
    ("broad", "What numerical modeling approaches are used for thermoelectric cooler design optimization?"),
    ("broad", "What machine learning approaches have been applied to thermoelectric cooler design?"),
    # --- specific numeric / edge cases ---
    ("edge", "What COP value has been achieved for thermoelectric coolers operating near room temperature?"),
    ("edge", "What is the maximum temperature difference achievable with a two-stage thermoelectric cooler?"),
    ("edge", "What voltage threshold causes COP to drop to zero in a thermoelectric cooler?"),
    ("edge", "What percentage improvement in COP has been reported from pulse-frequency modulation driving?"),
    ("edge", "What sample throughput has been achieved with thermoelectric-cooler-based PCR devices?"),
    ("edge", "What power consumption reduction has been achieved through thermoelectric cooler optimization?"),
    ("edge", "What accuracy has been reported for neural-network-based thermoelectric cooler performance prediction?"),
    ("edge", "What operating frequency range is used for pulse-frequency modulation of thermoelectric coolers?"),
    ("edge", "What lifetime or reliability data exists for thermoelectric coolers in continuous operation?"),
    ("edge", "What cost considerations affect thermoelectric cooler adoption compared to alternative cooling methods?"),
    # --- Thomson effect / advanced physics ---
    ("advanced", "What is the Thomson effect and how does it affect thermoelectric cooler modeling?"),
    ("advanced", "How is hyperbolic heat conduction relevant to micro-scale thermoelectric coolers?"),
    ("advanced", "What role does the Peltier effect play in thermoelectric cooler operation?"),
    ("advanced", "How does Joule heating affect thermoelectric cooler efficiency?"),
    ("advanced", "What temperature-dependent effects are important for accurate thermoelectric cooler models?"),
    ("advanced", "How is the figure of merit ZT related to thermoelectric cooler design optimization?"),
    ("advanced", "What role do contact interfaces play in thermoelectric device thermal resistance?"),
    ("advanced", "What uncertainty and sensitivity analysis methods have been applied to thermoelectric refrigerators?"),
]


def main() -> None:
    assert len(QUERIES) == 100, f"expected 100 queries, got {len(QUERIES)}"

    out_path = Path("scripts/_regression_probe_100_results.json")
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text())
    done_queries = {r["query"] for r in results}

    for i, (category, query) in enumerate(QUERIES, 1):
        if query in done_queries:
            continue
        print(f"\n[{i}/100] ({category}) {query}")
        t0 = time.time()
        try:
            response, trace = build_priors_response_with_trace(query)
            elapsed = time.time() - t0
            retries = len(trace.attempts) - 1
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
            }
            print(f"  -> {len(response.priors)} priors, {retries} retries, {elapsed:.0f}s")
        except Exception as e:  # noqa: BLE001 - want to capture and continue past any single failure
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
            }
            print(f"  -> HARD FAILURE: {type(e).__name__}: {e}")

        results.append(entry)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n\nDone. {len(results)} results saved to {out_path}")
    n_errors = sum(1 for r in results if r["error"])
    n_zero_priors = sum(1 for r in results if r["error"] is None and len(r["priors"]) == 0)
    total_retries = sum(r["retries"] or 0 for r in results)
    print(f"hard failures: {n_errors}  |  zero-prior results: {n_zero_priors}  |  total retries across all queries: {total_retries}")


if __name__ == "__main__":
    main()
