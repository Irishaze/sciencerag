"""04_sobol_design.py — Generate feasible Sobol samples and split into train/cal/val/test.

CRITICAL: Feasibility filtering happens BEFORE splitting.
Candidate pool auto-expands until 256 feasible points are found.

pitch constraint direction TBD — must verify against COMSOL definition.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats.qmc import Sobol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    PARAM_NAMES,
    PARAM_LOWER,
    PARAM_UPPER,
    PARAM_UNITS,
    N_PARAMS,
    SOBOL_SEED,
    N_TRAIN,
    N_CALIBRATION,
    N_VAL,
    N_TEST,
    DATA,
    check_geometric_feasibility,
)

# We need 256 feasible points
TARGET_N = N_TRAIN + N_CALIBRATION + N_VAL + N_TEST  # 256


def normalize_to_01(phys):
    """Normalize physical parameters to [0,1]."""
    return (phys - np.array(PARAM_LOWER)) / (np.array(PARAM_UPPER) - np.array(PARAM_LOWER))


def denormalize_to_phys(norm):
    """Denormalize [0,1] to physical units."""
    return norm * (np.array(PARAM_UPPER) - np.array(PARAM_LOWER)) + np.array(PARAM_LOWER)


def is_feasible(phys):
    """Check if a physical parameter vector is feasible."""
    # Check parameter bounds
    lo = np.array(PARAM_LOWER)
    hi = np.array(PARAM_UPPER)
    if np.any(phys < lo) or np.any(phys > hi):
        return False
    # Check geometric constraints
    return check_geometric_feasibility(phys)


def generate_feasible_sobol(n_needed=256, max_attempts=100):
    """Generate feasible Sobol samples, expanding candidate pool as needed."""
    feasible = []
    feasible_norm = []
    seed_offset = 0

    while len(feasible) < n_needed and seed_offset < max_attempts:
        current_seed = SOBOL_SEED + seed_offset
        # Generate a large batch
        m = max(9, int(np.ceil(np.log2(n_needed * (seed_offset + 2)))))
        n_batch = 2 ** m

        print(f"  Generating {n_batch} Sobol candidates (seed={current_seed}, m={m})...")
        sampler = Sobol(d=N_PARAMS, scramble=True, seed=current_seed)
        candidates_norm = sampler.random_base2(m=m)
        candidates_phys = denormalize_to_phys(candidates_norm)

        for i, phys in enumerate(candidates_phys):
            if is_feasible(phys):
                feasible.append(phys)
                feasible_norm.append(candidates_norm[i])
                if len(feasible) >= n_needed:
                    break

        print(f"    Feasible so far: {len(feasible)}/{len(candidates_phys)} "
              f"({100*len(feasible)/len(candidates_phys):.1f}%)")
        seed_offset += 1

    if len(feasible) < n_needed:
        raise RuntimeError(
            f"Could not find {n_needed} feasible points after {max_attempts} attempts. "
            f"Found only {len(feasible)}. Check geometric constraints."
        )

    return np.array(feasible[:n_needed]), np.array(feasible_norm[:n_needed])


def main():
    print(f"Target: {TARGET_N} feasible Sobol points")
    print(f"Parameters: {N_PARAMS} — {PARAM_NAMES}")
    print(f"Bounds: {list(zip(PARAM_LOWER, PARAM_UPPER))}")
    print()

    # Generate feasible points
    phys, norm = generate_feasible_sobol(n_needed=TARGET_N)
    print(f"\nGenerated {len(phys)} feasible points")

    # Check feasibility rate
    total_checked = len(phys)
    feasible_rate = len(phys) / max(total_checked, len(phys))
    print(f"Feasibility rate: {feasible_rate:.1%}")

    # Split
    train_phys = phys[:N_TRAIN]
    train_norm = norm[:N_TRAIN]
    calibration_phys = phys[N_TRAIN:N_TRAIN + N_CALIBRATION]
    calibration_norm = norm[N_TRAIN:N_TRAIN + N_CALIBRATION]
    val_phys = phys[N_TRAIN + N_CALIBRATION:N_TRAIN + N_CALIBRATION + N_VAL]
    val_norm = norm[N_TRAIN + N_CALIBRATION:N_TRAIN + N_CALIBRATION + N_VAL]
    test_phys = phys[N_TRAIN + N_CALIBRATION + N_VAL:]
    test_norm = norm[N_TRAIN + N_CALIBRATION + N_VAL:]

    print(f"\nSplit sizes:")
    print(f"  train:       {len(train_phys)}")
    print(f"  calibration: {len(calibration_phys)}")
    print(f"  val:         {len(val_phys)}")
    print(f"  test:        {len(test_phys)}")

    # Validate that splits cover disjoint regions reasonably
    for label, data in [("train", train_phys), ("calibration", calibration_phys),
                         ("val", val_phys), ("test", test_phys)]:
        ranges = np.ptp(data, axis=0)
        print(f"\n  {label} param ranges: {dict(zip(PARAM_NAMES, [f'{r:.3g}' for r in ranges]))}")

    # Save
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(DATA / "sobol_256_phys.npy", phys)
    np.save(DATA / "sobol_256_norm.npy", norm)
    np.save(DATA / "split_train_phys.npy", train_phys)
    np.save(DATA / "split_train_norm.npy", train_norm)
    np.save(DATA / "split_calibration_phys.npy", calibration_phys)
    np.save(DATA / "split_calibration_norm.npy", calibration_norm)
    np.save(DATA / "split_val_phys.npy", val_phys)
    np.save(DATA / "split_val_norm.npy", val_norm)
    np.save(DATA / "split_test_phys.npy", test_phys)
    np.save(DATA / "split_test_norm.npy", test_norm)

    # Metadata
    metadata = {
        "n_params": N_PARAMS,
        "param_names": PARAM_NAMES,
        "param_lower": PARAM_LOWER,
        "param_upper": PARAM_UPPER,
        "param_units": PARAM_UNITS,
        "sobol_seed_base": SOBOL_SEED,
        "target_n": TARGET_N,
        "feasible_rate": float(feasible_rate),
        "splits": {
            "train": [0, N_TRAIN],
            "calibration": [N_TRAIN, N_TRAIN + N_CALIBRATION],
            "val": [N_TRAIN + N_CALIBRATION, N_TRAIN + N_CALIBRATION + N_VAL],
            "test": [N_TRAIN + N_CALIBRATION + N_VAL, TARGET_N],
        },
        "nested_subsets": {
            "64": [0, 64],
            "128": [0, 128],
            "160": [0, 160],
        },
    }
    with open(DATA / "sobol_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {DATA}/sobol_*.npy and sobol_metadata.json")
    print("Done.")


if __name__ == "__main__":
    main()
