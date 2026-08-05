"""Export the solved one-pair COMSOL model to component-graph field data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import mph

from physics_foundation.comsol_export import export_component_case

DEFAULT_MODEL = PROJECT / "models" / "tec_1pair.mph"
DEFAULT_OUTPUT = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"
DEFAULT_SUMMARY = PROJECT / "outputs" / "tec_1pair_component_export.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="dset3")
    parser.add_argument("--geometry-dataset", default="dset2")
    parser.add_argument("--cores", type=int, default=2)
    args = parser.parse_args()

    client = mph.start(cores=args.cores)
    try:
        model = client.load(str(args.model))
        summary = export_component_case(
            model,
            args.output,
            dataset_tag=args.dataset,
            geometry_dataset_tag=args.geometry_dataset,
        )
    finally:
        client.clear()

    DEFAULT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
