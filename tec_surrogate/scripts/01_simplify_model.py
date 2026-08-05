"""01_simplify_model.py — Simplify TEC model to 1 PN pair.

Model simplification: n_length=2, n_width=1, N=1 → 1 PN pair.
The geometry uses Array features parameterized by n_length/n_width/N.
"""

import shutil
import sys
import time
from pathlib import Path

import mph
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    SOURCE_MODEL,
    SIMPLIFIED_MODEL,
    WORKING_MODEL,
    MODELS_MPH,
)


def discover_studies(model):
    """Return dict mapping known labels to study tags."""
    studies = {}
    jm = model.java
    for tag in list(jm.study().tags()):
        study = jm.study(tag)
        label = str(study.label()) if hasattr(study, 'label') else str(tag)
        studies[label] = str(tag)
        print(f"  Study found: tag='{tag}', label='{label}'")
    return studies


def main():
    print("Starting COMSOL...")
    client = mph.start()

    # Load original model
    print(f"Loading: {SOURCE_MODEL}")
    model = client.load(str(SOURCE_MODEL))
    print(f"  Model: {model.name()}")

    # Discover study names before modification
    print("\nDiscovering existing studies:")
    studies = discover_studies(model)

    # Find a stationary study to test-solve
    # Use tags rather than labels (avoid encoding issues with Chinese chars)
    stationary_study = None
    # std2 = "功率和散热" (stationary without sweep — safest to test)
    # std3 = "温差 vs. 电流" (parametric sweep)
    # std4 = "制冷系数" (parametric sweep)
    # Prefer std2 (simple stationary) > std3 > std4 > std1 (optimization)
    for preferred_tag in ["std2", "std3", "std4", "std1"]:
        for label, tag in studies.items():
            if tag == preferred_tag:
                stationary_study = label
                break
        if stationary_study:
            break
    print(f"  Using stationary study for test: tag='{studies[stationary_study]}', "
          f"label='{stationary_study}'")

    # === Simplify ===
    print("\nSetting n_length=2, n_width=1, N=1...")
    model.parameter("n_length", "2")
    model.parameter("n_width", "1")
    model.parameter("N", "n_length*n_width/2")  # Make N formula depend on n_length*n_width

    N_val = model.parameter("N", evaluate=True)
    nlen_val = model.parameter("n_length", evaluate=True)
    nwid_val = model.parameter("n_width", evaluate=True)
    print(f"  n_length = {nlen_val}")
    print(f"  n_width  = {nwid_val}")
    print(f"  N        = {N_val}")
    assert N_val == 1.0, f"Expected N=1, got {N_val}"

    # === Rebuild geometry ===
    print("\nRebuilding geometry...")
    jm = model.java
    comp = jm.component("comp1")
    geom = comp.geom("geom1")
    geom.run()
    nd = geom.getNDomains()
    print(f"  Geometry rebuilt: {nd} domains")

    # === Remesh ===
    print("Remeshing...")
    mesh = comp.mesh("mesh1")
    mesh.run()
    try:
        ne = mesh.getNElements()
    except Exception:
        ne = -1
    print(f"  Mesh complete, elements: {ne}")

    # === Test solve ===
    print(f"\nTest solving ('{stationary_study}')...")
    started = time.perf_counter()
    try:
        model.solve(stationary_study)
        elapsed = time.perf_counter() - started
        print(f"  Solved in {elapsed:.1f}s")
    except Exception as e:
        print(f"  Solve failed: {str(e)[:200]}")
        print("  Trying to solve all studies...")
        for label in studies:
            try:
                model.solve(label)
                print(f"  '{label}' solved successfully")
                break
            except Exception:
                continue

    # === Save simplified model ===
    MODELS_MPH.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {SIMPLIFIED_MODEL}")
    # Remove old file first if it exists (fix file lock issue)
    if SIMPLIFIED_MODEL.exists():
        SIMPLIFIED_MODEL.unlink()
    model.save(str(SIMPLIFIED_MODEL))
    print(f"  Saved: {SIMPLIFIED_MODEL.stat().st_size / 1e6:.1f} MB")

    # Create working copy (fresh file, not locked)
    if WORKING_MODEL.exists():
        WORKING_MODEL.unlink()
    shutil.copy2(str(SIMPLIFIED_MODEL), str(WORKING_MODEL))
    print(f"  Working copy: {WORKING_MODEL}")

    # === Add transient study to working model ===
    print("\nAdding transient study...")
    model_work = client.load(str(WORKING_MODEL))
    jm_work = model_work.java

    try:
        existing = set(jm_work.study().tags())
        if "std5" not in existing:
            study5 = jm_work.study().create("std5")
            study5.label("Transient COP")
            step = study5.create("tstep1", "Transient")
            step.label("Time Dependent")
            try:
                step.set("timelist", "range(0,1,60)")
            except Exception:
                try:
                    step.set("tlist", "range(0,1,60)")
                except Exception:
                    pass
            print("  Created transient study 'Transient COP' (std5)")
        else:
            print("  Transient study already exists")
    except Exception as e:
        print(f"  Failed to create transient study: {e}")
        print("  The model has stationary-only studies. Transient will need manual COMSOL GUI setup.")
        print("  For Phase 1 pipeline verification, we proceed with stationary data only.")

    # Save working model (need to clear client first to release file lock)
    model_work.save(str(WORKING_MODEL))
    print(f"  Working model saved: {WORKING_MODEL.stat().st_size / 1e6:.1f} MB")

    print("\n=== Simplification complete ===")
    print(f"  Simplified model: {SIMPLIFIED_MODEL}")
    print(f"  Working model:    {WORKING_MODEL}")
    print(f"  Domains: {nd}")

    # Save study info for later use
    import json
    with open(MODELS_MPH / "study_info.json", "w", encoding="utf-8") as f:
        json.dump(studies, f, ensure_ascii=False, indent=2)
    print(f"  Study info saved to: {MODELS_MPH / 'study_info.json'}")

    client.clear()


if __name__ == "__main__":
    main()
