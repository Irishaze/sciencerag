"""03_define_probes.py — Define normalized probe coordinates for spatial field extraction.

Generates a fixed set of probe points in normalized coordinates (ξ,η,ζ) ∈ [0,1]³
for each region. These are stored once and reused across all samples.

For each sample, physical coordinates are computed by:
  x_phys = ξ * (x_max_region - x_min_region) + x_min_region
(same for y, z)

The probe definitions are saved as a structured .npz for use by 05_run_batch.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    REGIONS,
    COMPONENT_ROLES,
    REGION_TO_COMPONENT,
    N_REGION_TYPES,
    N_COMPONENT_ROLES,
    DATA,
)

# === Physical layout for nominal TEC (1 pair, n_length=2, n_width=1) ===
# Used to define region bounding boxes in physical coordinates.
# These will be recomputed per-sample based on actual parameters during batch runs.
# The normalized coords are universal.

# For reference, nominal dimensions:
#   leg_length=1mm, leg_width=1.2mm, pitch=0.5mm
#   d_conductor=100um=0.1mm, d_ceramics=0.3mm
#   network_length = 2*1 + 0.5 = 2.5mm
#   network_width  = 1.2mm
#   leg_height = 2.5 - 2*(0.1+0.3) = 1.7mm


def generate_region_probes(region_name, grid, interface_offset_factor=0.01):
    """Generate normalized probe points for a single region.

    Args:
        region_name: key in REGIONS
        grid: (nx, ny, nz) tuple
        interface_offset_factor: for interface-offset variants

    Returns:
        probes: (N, 3) array of (ξ, η, ζ) in [0,1]
        volumes: (N,) per-point volume weights
        types: (N,) 'internal' or 'interface'
    """
    nx, ny, nz = grid
    region_info = REGIONS[region_name]

    # Internal grid points (cell centers)
    xi = (np.arange(nx) + 0.5) / nx
    eta = (np.arange(ny) + 0.5) / ny
    zeta = (np.arange(nz) + 0.5) / nz
    XX, YY, ZZ = np.meshgrid(xi, eta, zeta, indexing="ij")
    internal = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Volume weight: each cell = 1/(nx*ny*nz) of region volume
    n_internal = internal.shape[0]
    volumes = np.full(n_internal, 1.0 / n_internal)
    types = np.array(["internal"] * n_internal)

    # Interface offset points (z-faces only for simplicity)
    interface_points = []
    for face_z in [0.0, 1.0]:
        # Offset slightly inward from face
        offset = interface_offset_factor
        z = face_z + offset * (1 if face_z == 0 else -1)
        z = np.clip(z, 0.01, 0.99)

        # Coarse grid on face
        fxi = (np.arange(max(2, nx // 2)) + 0.5) / max(2, nx // 2)
        feta = (np.arange(max(2, ny // 2)) + 0.5) / max(2, ny // 2)
        fXX, fYY = np.meshgrid(fxi, feta, indexing="ij")
        for x, y in zip(fXX.ravel(), fYY.ravel()):
            interface_points.append([x, y, z])

    if interface_points:
        interface = np.array(interface_points)
        n_intf = interface.shape[0]
        all_points = np.vstack([internal, interface])
        all_volumes = np.concatenate([volumes, np.zeros(n_intf)])
        all_types = np.concatenate([types, np.array(["interface"] * n_intf)])
    else:
        all_points = internal
        all_volumes = volumes
        all_types = types

    return all_points, all_volumes, all_types


def compute_region_bbox_nominal(region_name):
    """Compute nominal physical bounding box for a region.

    These serve as reference but will be recomputed per-sample.
    We just need them to validate the approach.
    """
    # Nominal parameters
    L, W, H = 8.0, 10.0, 2.5      # mm
    d_cond = 0.1                    # mm
    d_cer = 0.3                     # mm
    ll, lw = 1.0, 1.2              # leg_length, leg_width
    pitch = 0.5
    leg_h = H - 2 * (d_cond + d_cer)  # 1.7mm
    nw_len = 2 * ll + pitch         # network_length = 2.5mm
    nw_wid = lw                     # network_width = 1.2mm

    # Regions stack in z: cold_ceramic → cold_conductor → legs → hot_conductor → hot_ceramic
    z_base = -H / 2
    z_cold_cer = z_base
    z_cold_cond = z_cold_cer + d_cer
    z_leg_bottom = z_cold_cond + d_cond
    z_leg_top = z_leg_bottom + leg_h
    z_hot_cond = z_leg_top
    z_hot_cer = z_hot_cond + d_cond

    bboxes = {
        "cold_ceramic": {
            "x_min": -L / 2, "x_max": L / 2,
            "y_min": -W / 2, "y_max": W / 2,
            "z_min": z_cold_cer, "z_max": z_cold_cer + d_cer,
        },
        "cold_conductor": {
            "x_min": -nw_len / 2, "x_max": nw_len / 2,
            "y_min": -nw_wid / 2, "y_max": nw_wid / 2,
            "z_min": z_cold_cond, "z_max": z_cold_cond + d_cond,
        },
        "p_leg": {
            "x_min": -0.5, "x_max": 0.0,  # left leg (approximate)
            "y_min": -lw / 2, "y_max": lw / 2,
            "z_min": z_leg_bottom, "z_max": z_leg_top,
        },
        "n_leg": {
            "x_min": 0.0, "x_max": 0.5,   # right leg (approximate)
            "y_min": -lw / 2, "y_max": lw / 2,
            "z_min": z_leg_bottom, "z_max": z_leg_top,
        },
        "hot_conductor": {
            "x_min": -nw_len / 2, "x_max": nw_len / 2,
            "y_min": -nw_wid / 2, "y_max": nw_wid / 2,
            "z_min": z_hot_cond, "z_max": z_hot_cond + d_cond,
        },
        "hot_ceramic": {
            "x_min": -L / 2, "x_max": L / 2,
            "y_min": -W / 2, "y_max": W / 2,
            "z_min": z_hot_cer, "z_max": z_hot_cer + d_cer,
        },
    }
    return bboxes[region_name]


def main():
    all_xi_eta_zeta = []
    all_region_onehot = []
    all_component_onehot = []
    all_volumes = []
    all_types = []
    all_region_names = []
    all_component_names = []

    for region_name, region_info in REGIONS.items():
        grid = region_info["grid"]
        probes, vols, types = generate_region_probes(region_name, grid)
        n = len(probes)

        # One-hot for region type
        region_idx = region_info["id"]
        region_oh = np.zeros((n, N_REGION_TYPES))
        region_oh[:, region_idx] = 1.0

        # One-hot for component role — use first role for this region
        roles = REGION_TO_COMPONENT[region_name]
        comp_oh = np.zeros((n, N_COMPONENT_ROLES))
        comp_idx = COMPONENT_ROLES.index(roles[0])
        comp_oh[:, comp_idx] = 1.0

        all_xi_eta_zeta.append(probes)
        all_region_onehot.append(region_oh)
        all_component_onehot.append(comp_oh)
        all_volumes.append(vols)
        all_types.append(types)
        all_region_names.extend([region_name] * n)
        all_component_names.extend([roles[0]] * n)

        print(f"  {region_name:20s}: {n:4d} points "
              f"({np.sum(types == 'internal')} internal, "
              f"{np.sum(types == 'interface')} interface)")

    # Concatenate
    xez = np.vstack(all_xi_eta_zeta)
    region_oh = np.vstack(all_region_onehot)
    comp_oh = np.vstack(all_component_onehot)
    volumes = np.concatenate(all_volumes)
    types = np.concatenate(all_types)

    total = len(xez)
    n_internal = int(np.sum(types == "internal"))
    n_interface = int(np.sum(types == "interface"))

    print(f"\nTotal: {total} probe points ({n_internal} internal, {n_interface} interface)")

    # Compute nominal physical coordinates for reference
    print("\nNominal physical bounding boxes (reference only):")
    region_bboxes = {}
    for region_name in REGIONS:
        bbox = compute_region_bbox_nominal(region_name)
        region_bboxes[region_name] = bbox
        print(f"  {region_name}: x=[{bbox['x_min']:.3f},{bbox['x_max']:.3f}], "
              f"y=[{bbox['y_min']:.3f},{bbox['y_max']:.3f}], "
              f"z=[{bbox['z_min']:.3f},{bbox['z_max']:.3f}]")

    # Save
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA / "probe_definitions.npz",
        xi_eta_zeta=xez,
        region_onehot=region_oh,
        component_onehot=comp_oh,
        volumes=volumes,
        types=types,
        region_names=np.array(all_region_names),
        component_names=np.array(all_component_names),
        n_regions=N_REGION_TYPES,
        n_components=N_COMPONENT_ROLES,
        n_internal=n_internal,
        n_interface=n_interface,
        n_total=total,
    )
    # Also save region bboxes for reference
    np.savez(DATA / "region_bboxes_nominal.npz", **{
        k: np.array([v["x_min"], v["x_max"], v["y_min"], v["y_max"], v["z_min"], v["z_max"]])
        for k, v in region_bboxes.items()
    })

    print(f"\nSaved probe definitions to: {DATA / 'probe_definitions.npz'}")
    print(f"Saved nominal bboxes to: {DATA / 'region_bboxes_nominal.npz'}")


if __name__ == "__main__":
    main()
