"""4.1 anomaly checks (spec §4.1): ood.

Each check function always returns an Anomaly — including a passing check,
at severity="info" — rather than only reporting failures. That keeps the
response an auditable record of what was actually checked (spec §2 "凡论断
必有来源"), not just a silent absence when nothing went wrong.

energy_balance/pde_residual used to live here too, backed by tec_surrogate's
compositional multi-pair model — removed because that model was never a
substitute for real multi-pair COMSOL calibration data (its own training
script said so), so the two checks were reporting a residual against an
approximation with no real physical grounding rather than doing anything
"ood" doesn't already cover honestly.
"""

from __future__ import annotations

import numpy as np

from sciencerag.validate import tec_bridge
from sciencerag.validate.models import Anomaly, ValidateRequest


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
    return [check_ood(request)]
