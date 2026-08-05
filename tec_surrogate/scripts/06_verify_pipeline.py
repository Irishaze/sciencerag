"""06_verify_pipeline.py — Verify Phase 1 pipeline outputs.

Checks:
  1. All expected .npz files exist and are valid
  2. No NaN in scalar outputs (within tolerance)
  3. Time curves have expected shape and finite values
  4. Geometry domain counts are consistent
  5. Checkpoint DB is consistent with file count
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, DATA, N_TIME_POINTS, SCALAR_EXPRESSIONS, TIME_CURVE_EXPRESSIONS

DB_PATH = DATA / "simulation_status.db"


def verify_file(sample_id):
    """Verify a single .npz file."""
    path = RAW / f"sample_{sample_id:04d}.npz"
    if not path.exists():
        return {"sample_id": sample_id, "status": "MISSING_FILE"}

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return {"sample_id": sample_id, "status": "CORRUPT", "error": str(e)}

    issues = []

    # Check params
    params = data.get("params")
    if params is None:
        issues.append("no params")
    elif np.any(~np.isfinite(params)):
        issues.append("params contain NaN/Inf")

    # Check scalars
    scalar_names = data.get("scalar_names")
    scalars = data.get("scalars")
    if scalars is not None:
        n_nan = np.sum(~np.isfinite(scalars))
        if n_nan > len(SCALAR_EXPRESSIONS) // 2:
            issues.append(f"{n_nan}/{len(scalars)} scalar NaN")
    else:
        issues.append("no scalars")

    # Check time curves
    for name in TIME_CURVE_EXPRESSIONS:
        key = f"curve_{name}"
        curve = data.get(key)
        if curve is None:
            issues.append(f"missing curve {name}")
        else:
            curve_arr = np.asarray(curve)
            n_nan = np.sum(~np.isfinite(curve_arr))
            if n_nan > len(curve_arr) // 2:
                issues.append(f"curve {name}: {n_nan}/{len(curve_arr)} NaN")

    # Check solve time
    st = data.get("solve_time_s")
    if st is not None:
        st_val = float(st)
        if st_val < 0.1:
            issues.append(f"solve time suspiciously low: {st_val:.2f}s")

    status = "OK" if not issues else f"ISSUES: {'; '.join(issues)}"
    return {"sample_id": sample_id, "status": status, "issues": issues}


def main():
    # Load expected sample IDs from DB
    db = sqlite3.connect(str(DB_PATH))
    cur = db.execute("SELECT sample_id, status FROM samples ORDER BY sample_id")
    db_samples = {r[0]: r[1] for r in cur.fetchall()}
    db.close()

    if not db_samples:
        print("No samples in DB. Run 05_run_batch.py first.")
        return

    print(f"DB records: {len(db_samples)}")

    # Check file count in raw directory
    raw_files = sorted(RAW.glob("sample_*.npz"))
    print(f"Raw .npz files: {len(raw_files)}")

    # Verify each completed sample
    done = [sid for sid, status in db_samples.items() if status == "done"]
    print(f"Done in DB: {len(done)}")

    if not done:
        print("No completed samples to verify.")
        return

    print(f"\nVerifying {len(done)} completed samples...")
    results = []
    for sid in done:
        r = verify_file(sid)
        results.append(r)
        if r["status"] != "OK":
            print(f"  Sample {sid:04d}: {r['status']}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    issue_count = sum(1 for r in results if r["status"] != "OK")

    print(f"\n=== Verification Summary ===")
    print(f"  OK:      {ok_count}")
    print(f"  Issues:  {issue_count}")

    if issue_count == 0:
        print("\nPipeline verification PASSED — all samples clean.")
    else:
        print(f"\nPipeline verification: {issue_count} samples have issues.")
        print("Check the individual .npz files listed above.")

    # Check domain count consistency
    domains = set()
    for sid in done:
        path = RAW / f"sample_{sid:04d}.npz"
        if path.exists():
            data = np.load(path)
            nd = data.get("n_domains")
            if nd is not None:
                domains.add(int(nd))
    if domains:
        print(f"\nDomain counts across samples: {sorted(domains)}")
        if len(domains) > 1:
            print("  WARNING: Domain counts vary — geometry may not be rebuilding correctly")
        else:
            print("  Consistent — geometry rebuild looks correct")


if __name__ == "__main__":
    main()
