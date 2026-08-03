"""4.1 anomaly checks (spec §4.1): energy_balance, pde_residual, ood.

Each check function always returns an Anomaly — including a passing check,
at severity="info" — rather than only reporting failures. That keeps the
response an auditable record of what was actually checked (spec §2 "凡论断
必有来源"), not just a silent absence when nothing went wrong.
"""

from __future__ import annotations

import numpy as np

from sciencerag.validate import tec_bridge
from sciencerag.validate.models import Anomaly, ValidateRequest


def _severity_from_baseline(value: float, baseline: tuple[float, ...]) -> tuple[str, dict]:
    reference = np.asarray(baseline, dtype=float)
    ref_mean = float(reference.mean())
    ref_max = float(reference.max())
    ratio = float("inf") if ref_max <= 0 and value > 0 else (value / ref_max if ref_max > 0 else 0.0)
    if ratio > 5:
        severity = "blocking"
    elif ratio > 2:
        severity = "warning"
    else:
        severity = "info"
    evidence = {
        "residual_total": value,
        "baseline_mean": ref_mean,
        "baseline_max": ref_max,
        "ratio_to_baseline_max": ratio,
        "baseline_note": (
            "baseline = same residual computed on the 11 known solved "
            "one-pair COMSOL operating points; absolute scale is not "
            "calibrated (tec_surrogate PHYSICS_FOUNDATION.md), only the "
            "ratio to this baseline is meaningful"
        ),
    }
    return severity, evidence


def check_energy_balance(request: ValidateRequest) -> Anomaly:
    if request.field_case_index is None:
        return Anomaly(
            check="energy_balance",
            severity="info",
            evidence={
                "skipped": True,
                "reason": (
                    "no field_case_index given: conservation check needs a "
                    "solved field case with real interface data, which today "
                    "only exists for the 11 bundled one-pair COMSOL operating "
                    "points"
                ),
            },
        )
    if request.n_pairs != 1:
        return Anomaly(
            check="energy_balance",
            severity="info",
            evidence={
                "skipped": True,
                "reason": (
                    "conservation check only covers n_pairs=1: composing to "
                    "n_pairs>1 (compose_multi_pair_graph) invents new "
                    "module-level interfaces with no matched real interface "
                    "samples to check conservation against"
                ),
                "n_pairs": request.n_pairs,
            },
        )
    value = tec_bridge.conservation_residual_total(request.field_case_index)
    baseline = tec_bridge.conservation_baseline()
    severity, evidence = _severity_from_baseline(value, baseline)
    evidence["field_case_index"] = request.field_case_index
    evidence["terms"] = tec_bridge.conservation_residual(request.field_case_index)
    return Anomaly(check="energy_balance", severity=severity, evidence=evidence)


def check_pde_residual(request: ValidateRequest) -> Anomaly:
    if request.field_case_index is None:
        return Anomaly(
            check="pde_residual",
            severity="info",
            evidence={
                "skipped": True,
                "reason": (
                    "no field_case_index given: PDE residual needs a solved "
                    "one-pair operating point to compose from"
                ),
            },
        )
    value = tec_bridge.pde_residual_total(request.field_case_index, request.n_pairs)
    baseline = tec_bridge.pde_residual_baseline(request.n_pairs)
    severity, evidence = _severity_from_baseline(value, baseline)
    evidence["field_case_index"] = request.field_case_index
    evidence["n_pairs"] = request.n_pairs
    evidence["terms"] = tec_bridge.pde_residual(request.field_case_index, request.n_pairs)
    evidence["calibration"] = (
        "one_pair_comsol_anchor" if request.n_pairs == 1
        else "composed_topology_pending_multipair_comsol"
    )
    return Anomaly(check="pde_residual", severity=severity, evidence=evidence)


def check_ood(request: ValidateRequest) -> Anomaly:
    if request.latent_state is None:
        return Anomaly(
            check="ood",
            severity="info",
            evidence={
                "skipped": True,
                "reason": "no latent_state given: OOD check needs 06's z output",
            },
        )
    training_z = tec_bridge.load_training_latent()
    expected_dim = training_z.shape[1]
    if len(request.latent_state) != expected_dim:
        return Anomaly(
            check="ood",
            severity="warning",
            evidence={
                "error": (
                    f"latent_state has {len(request.latent_state)} dims, "
                    f"expected {expected_dim} (comsol_latent_surrogate.joblib "
                    "latent_dim) — cannot score, treated as an anomaly in "
                    "itself rather than silently skipped"
                ),
            },
        )
    z = np.asarray(request.latent_state, dtype=float)
    mean = training_z.mean(axis=0)
    std = training_z.std(axis=0, ddof=1)
    std = np.where(std > 1e-8, std, 1.0)
    # PCA scores are uncorrelated by construction, so a per-axis z-score sum
    # of squares is the Mahalanobis distance without needing to invert a
    # (near-singular, n=31 vs 5 dims) empirical covariance matrix.
    per_axis_z = (z - mean) / std
    mahalanobis = float(np.sqrt(np.sum(per_axis_z**2)))
    # Reference distribution: same score computed for every training point
    # against the leave-one-out mean/std of the rest — gives a real
    # empirical distribution of "how OOD does a known-in-distribution point
    # look" to compare this run against, instead of a chi-square assumption
    # that training_z's small sample (n=31) may not satisfy.
    training_scores = []
    n = len(training_z)
    for index in range(n):
        rest = np.delete(training_z, index, axis=0)
        rest_mean = rest.mean(axis=0)
        rest_std = rest.std(axis=0, ddof=1)
        rest_std = np.where(rest_std > 1e-8, rest_std, 1.0)
        score = np.sqrt(np.sum(((training_z[index] - rest_mean) / rest_std) ** 2))
        training_scores.append(float(score))
    training_scores.sort()
    percentile = float(np.searchsorted(training_scores, mahalanobis) / n * 100)
    if percentile >= 99:
        severity = "blocking"
    elif percentile >= 90:
        severity = "warning"
    else:
        severity = "info"
    return Anomaly(
        check="ood",
        severity=severity,
        evidence={
            "mahalanobis_distance": mahalanobis,
            "training_sample_count": n,
            "training_distance_percentile": percentile,
            "training_distance_range": [training_scores[0], training_scores[-1]],
        },
    )


def run_anomaly_checks(request: ValidateRequest) -> list[Anomaly]:
    return [
        check_energy_balance(request),
        check_pde_residual(request),
        check_ood(request),
    ]
