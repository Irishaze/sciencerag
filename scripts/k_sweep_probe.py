"""M1 threshold-recalibration Step 1: retrieval top-k sweep.

Background: current evidence_k is paper-qa's unmodified default (10),
inherited from before the sim_params.json contract sync (spec §3.6) — it
was never tuned for "cover these 12 specific geometry parameters," and a
too-small k can starve extraction of evidence before MIN_EVIDENCE_RELEVANCE
or CONFIDENCE_THRESHOLD ever get a chance to matter (a false gap from the
retrieval gate, not the relevance/confidence gates — see spec §3.7).

For each evidence_k in K_VALUES, runs every query in QUERIES and takes the
UNION, across all queries, of which of the 12 sim_params.json geometry_free
parameters got at least one drafted prior (field or related_fields) BEFORE
the confidence filter — i.e. extraction-reachable coverage as a function of
k alone, holding MIN_EVIDENCE_RELEVANCE/CONFIDENCE_THRESHOLD out of the
picture. Report per-k coverage so a human can read off the elbow (the k
past which coverage stops improving) rather than guessing.

Self-contained by design: does NOT import or modify
sciencerag/priors/retrieval.py's build_settings()/run_query() or touch the
production evidence_k. Builds its own paper-qa Settings per k, reusing only
the shared building blocks (embedding/LLM config, evidence-table
construction, extraction) retrieval.py already exposes as a package-level
API. The production query path's behavior is unchanged by this script
existing or running.

Real API calls (embeddings + PaperQA2's agentic retrieval + DeepSeek
extraction) for every (k, query) pair — costs money and takes real time.
With the default 10 queries x 4 k-values = 40 real pipeline runs.

    uv run python scripts/k_sweep_probe.py
"""

import json
from pathlib import Path

from paperqa import Settings, ask

from sciencerag.common.config import get_embedding_model, get_llm_model
from sciencerag.priors.contract import GEOMETRY_FREE_NAMES
from sciencerag.priors.extract import extract_priors
from sciencerag.priors.retrieval import CORPUS_DIR, INDEX_DIR, _build_evidence_table

K_VALUES = [10, 20, 30, 50]

# One or two queries per sim_params.json geometry_free parameter (spec
# §3.6), reusing wording from tests/fixtures/priors_regression.json where
# it already targets the right parameter, so the union of these queries
# should touch all 12 if the corpus has anything on them at all.
QUERIES: list[str] = [
    "What is a typical leg length (leg_length) used in Bi2Te3 thermoelectric cooler design — range or optimal value?",
    "How does the optimal leg length (leg_length) depend on leg width (leg_width) in Bi2Te3 thermoelectric cooler design?",
    "What leg pitch or spacing (pitch) between thermoelectric legs is typically used in Bi2Te3 cooler design?",
    "What conductor/metallization layer thickness (d_conductor) is typically used in Bi2Te3 thermoelectric leg fabrication?",
    "What ceramic substrate thickness (d_ceramics) is typically used in thermoelectric cooler module design?",
    "What overall module dimensions (length, width, height) are typical for a Bi2Te3 thermoelectric cooler?",
    "What heat sink fin geometries (fin height, fin thickness, fin count) are reported to improve thermoelectric cooler performance?",
    "What heat sink base plate thickness is typically used in thermoelectric cooler heat sink design?",
    "What geometric design limits (leg length, leg width, leg spacing) constrain the maximum achievable COP in Bi2Te3 thermoelectric cooling?",
    "What thermoelectric cooler leg and heat sink geometry has been reported for battery thermal management applications?",
]

OUT_PATH = Path("data/k_sweep_results.json")


def _settings_for_k(k: int) -> Settings:
    llm = get_llm_model()
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=get_embedding_model(),
        paper_directory=str(CORPUS_DIR),
    )
    settings.answer.evidence_k = k
    settings.agent.index.index_directory = str(INDEX_DIR)
    return settings


def _drafted_params_for(query: str, k: int) -> tuple[set[str], dict]:
    """Returns (covered geometry_free names, raw detail dict for logging)."""
    response = ask(query, settings=_settings_for_k(k))
    contexts = response.session.contexts
    if not contexts:
        return set(), {"n_contexts": 0, "n_evidence": 0, "n_priors": 0, "error": None}

    evidence_table, _below_threshold_evidence = _build_evidence_table(contexts)
    if not evidence_table:
        return set(), {
            "n_contexts": len(contexts),
            "n_evidence": 0,
            "n_priors": 0,
            "error": None,
        }

    try:
        priors, _filtered_material_count = extract_priors(query, evidence_table)
    except Exception as e:  # noqa: BLE001 - keep the sweep going past a single bad call
        return set(), {
            "n_contexts": len(contexts),
            "n_evidence": len(evidence_table),
            "n_priors": 0,
            "error": f"{type(e).__name__}: {e}",
        }

    covered = set()
    for p in priors:
        if p.field:
            covered.add(p.field)
        covered.update(p.related_fields)
    covered &= GEOMETRY_FREE_NAMES

    return covered, {
        "n_contexts": len(contexts),
        "n_evidence": len(evidence_table),
        "n_priors": len(priors),
        "error": None,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(exist_ok=True)
    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
    done = {(r["k"], r["query"]) for r in results}

    for k in K_VALUES:
        for query in QUERIES:
            if (k, query) in done:
                continue
            print(f"\n[k={k}] {query}")
            covered, detail = _drafted_params_for(query, k)
            print(
                f"  -> {detail['n_contexts']} contexts, {detail['n_evidence']} evidence, "
                f"{detail['n_priors']} priors, covers {sorted(covered)}"
            )
            results.append({"k": k, "query": query, "covered": sorted(covered), **detail})
            OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n\n" + "=" * 70)
    print("Coverage of the 12 geometry_free parameters, union across all queries, by k")
    print("=" * 70)
    for k in K_VALUES:
        union_covered: set[str] = set()
        for r in results:
            if r["k"] == k:
                union_covered |= set(r["covered"])
        missing = sorted(GEOMETRY_FREE_NAMES - union_covered)
        print(f"k={k}: {len(union_covered)}/{len(GEOMETRY_FREE_NAMES)} covered  |  missing: {missing}")


if __name__ == "__main__":
    main()
