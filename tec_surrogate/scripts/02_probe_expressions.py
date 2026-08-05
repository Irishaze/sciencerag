"""02_probe_expressions.py — Discover valid expression names in the TEC model.

Tests candidate expressions against the model to confirm which COMSOL entity
names (aveop1, aveop2, ec.V0_5, etc.) actually work.
"""

import json
import sys
from pathlib import Path

import mph
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SIMPLIFIED_MODEL, OUTPUTS

# Candidate expressions to test (grouped by purpose)
CANDIDATES = {
    "temperature_operators": [
        "aveop1(T)",
        "aveop2(T)",
        "aveop1(T)-aveop2(T)",
    ],
    "circuit_terminals": [
        "ec.V0_5",
        "ec.I0_5",
        "ec.V0_1",
        "ec.I0_1",
        "ec.V0_2",
        "ec.I0_2",
        "ec.V0_3",
        "ec.I0_3",
        "ec.V0",
        "ec.I0",
    ],
    "heat_flux": [
        "ht.ntefluxInt",
        "ht.ntefluxMag",
        "ht.q0",
    ],
    "electric_field": [
        "ec.normE",
        "ec.normJ",
        "ec.Ex",
        "ec.Ey",
        "ec.Ez",
    ],
    "global_variables": [
        "Iset",
        "Vset",
        "Ufan",
        "tauI",
        "N",
        "Tref",
        "dT0",
        "I0",
    ],
}

DATASETS_TO_TEST = [
    "研究 2：功率和散热//解 2",
    "研究 3：温差 vs. 电流//解 3",
    "研究 4：制冷系数//解 4",
]


def try_evaluate(model, expr, dataset=None, **kwargs):
    """Try to evaluate an expression; return (success, preview, error_msg)."""
    try:
        val = model.evaluate(expr, dataset=dataset, **kwargs)
        arr = np.asarray(np.real_if_close(val, tol=1000), dtype=float).ravel()
        if arr.size == 0:
            return True, "<empty>", None
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return True, f"all {arr.size} values NaN/Inf", None
        preview = f"shape={arr.shape}, min={finite.min():.4g}, max={finite.max():.4g}, mean={finite.mean():.4g}"
        return True, preview, None
    except Exception as e:
        return False, None, str(e).splitlines()[0][:200]


def main():
    client = mph.start()
    model = client.load(str(SIMPLIFIED_MODEL))
    print(f"Model: {model.name()}\n")

    report = {"model": str(SIMPLIFIED_MODEL), "results": {}}

    # Test datasets
    print("=== Available Datasets ===")
    for ds in model.datasets():
        print(f"  {ds}")
        try:
            inner_idx, inner_vals = model.inner(ds)
            print(f"    inner: {len(inner_vals)} values")
        except Exception:
            print(f"    (no inner values)")

    print()

    # Test each candidate expression
    for group_name, expressions in CANDIDATES.items():
        print(f"--- {group_name} ---")
        report["results"][group_name] = []

        for expr in expressions:
            # Try without dataset first
            ok, preview, err = try_evaluate(model, expr)
            dataset_used = None

            if not ok:
                # Try with each dataset
                for ds in DATASETS_TO_TEST:
                    ok, preview, err = try_evaluate(model, expr, dataset=ds)
                    if ok:
                        dataset_used = ds
                        break

            status = "OK" if ok else "FAIL"
            ds_str = f" [{dataset_used}]" if dataset_used else ""
            print(f"  {status:4s}  {expr:40s}  {preview or err}{ds_str}")

            report["results"][group_name].append({
                "expression": expr,
                "success": ok,
                "dataset": dataset_used,
                "preview": preview,
                "error": err,
            })

    # Save report
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS / "expression_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nReport saved to: {report_path}")

    client.clear()


if __name__ == "__main__":
    main()
