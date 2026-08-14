"""Generate (or regenerate) the knowledge-graph's AI-designed entity/relation
type ontology and write it to data/kg/ontology.json.

Real LLM call, real cost — run explicitly, never from a hot request path:

    uv run python scripts/generate_kg_ontology.py
    uv run python scripts/generate_kg_ontology.py --force   # overwrite an existing ontology.json

Run this once before scripts/approve_kg_candidates.py's classification
step has anything to classify against, and again whenever sim_params.json
(the simulation contract this reads) meaningfully changes shape.
"""

from __future__ import annotations

import argparse
import sys

from sciencerag.priors.ontology_generator import ONTOLOGY_PATH, generate_ontology, save_ontology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing ontology.json")
    args = parser.parse_args()

    if ONTOLOGY_PATH.exists() and not args.force:
        print(f"{ONTOLOGY_PATH} already exists — pass --force to regenerate", file=sys.stderr)
        sys.exit(1)

    print("Generating ontology (one LLM call)...")
    ontology = generate_ontology()
    save_ontology(ontology)

    print(f"\nWrote {ONTOLOGY_PATH}\n")
    print(f"entity_types ({len(ontology.entity_types)}):")
    for t in ontology.entity_types:
        marker = " [fallback]" if t.is_fallback else ""
        print(f"  - {t.name}{marker}: {t.description}")
    print(f"\nedge_types ({len(ontology.edge_types)}):")
    for e in ontology.edge_types:
        print(f"  - {e.name}: {e.source} -> {e.target} ({e.description})")
    print(f"\n{ontology.analysis_summary}")


if __name__ == "__main__":
    main()
