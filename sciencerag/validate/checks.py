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

# blocking used to fire at the empirical 99th percentile of the 31-sample
# leave-one-out reference distribution. Verified against real data (2026-08)
# that this number has no real statistical grounding at n=31:
#   - resolving a genuine 99th percentile needs ~100 samples for even one
#     expected point beyond it, ~500 for a stable estimate — we have 31.
#   - the parametric alternative (treating the Mahalanobis-style statistic as
#     chi-square, df=5, which is what the math actually assumes) predicts a
#     99th-percentile distance of ~3.88 — but 2/31 (6.5%) of real, valid
#     training designs already exceed that, 6x the intended 1% rate. The real
#     latent distribution has a fatter tail than a Gaussian model predicts,
#     so borrowing the textbook chi-square cutoff isn't safe either.
# What n=31 *can* support: knowing the single most extreme point we've ever
# actually validated. BLOCKING_MARGIN scales that as a "worse than anything
# we've confirmed the surrogate can meaningfully represent" rule.
#
# 2026-08-17 first-principles re-derivation, margin dropped 1.5 -> 1.0:
# the original "with headroom so one historical outlier point doesn't
# singlehandedly define the boundary" justification didn't actually hold up
# under scrutiny — blocking_threshold = training_max * BLOCKING_MARGIN is a
# straight linear function of that single point (training_max) no matter
# what the multiplier is; a >1.0 margin never protected against the
# boundary being anchored to one n=1 observation, it only ever buffered
# "requests almost exactly as extreme as the worst confirmed case" from
# tripping the boundary on noise. Worse, extending trust 50% past the most
# extreme point we've ever actually validated is in real tension with what
# an OOD gate is *for* — the whole point is to not extrapolate past
# validated territory, and a >1.0 margin was doing exactly that. 1.0 means
# blocking now fires only on distances more extreme than anything we've
# confirmed the surrogate can represent, full stop — the most honest
# reading of "out of distribution" available at this sample size. Kept as a
# named constant (not inlined as training_max directly) so a future margin,
# if real evidence ever justifies one, stays a one-line change.
#
# Provisional either way: switch to a real empirical percentile once
# training data grows past ~100-150 samples.
BLOCKING_MARGIN = 1.0


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
    # training_distance_percentile is still computed and reported below
    # (useful audit context — "where does this land among 31 known-normal
    # designs") but is no longer what decides severity; see warning_threshold
    # below for why.
    percentile = float(np.searchsorted(training_scores, mahalanobis) / n * 100)
    training_max = training_scores[-1]
    # 2026-08-17: warning changed from percentile>=90-by-rank to the
    # midpoint of the data's own largest gap (excluding the top point,
    # which blocking already owns) after looking at the actual spacing
    # between the 31 sorted leave-one-out scores, not just their rank. The
    # real distribution isn't a smooth tail: most of the 31 scores climb in
    # tiny ~0.05-0.4 steps with no real structure, then there's a real
    # ~0.68 jump up to the 2nd-highest score, then a dominant ~8.8 jump up
    # to the max.
    #
    # An earlier version of this anchored warning directly to the
    # 2nd-highest score's own *value* (training_scores[-2]) rather than to
    # the gap itself — a smaller mistake of the same shape as the one
    # BLOCKING_MARGIN's docstring above describes: it put the boundary
    # exactly on top of a single real historical point again, so a new
    # request landing almost exactly on that point (a near-duplicate of
    # that one training design) sits right at the edge instead of clearly
    # on one side of it. Sitting anywhere *inside* the actual gap avoids
    # that — the training data classifies identically no matter where in
    # the gap the line falls, so the midpoint is the natural, symmetric
    # choice, not an arbitrary one.
    #
    # Computed generally (largest gap among all-but-the-top score, not a
    # position hardcoded to "index -2") so this keeps finding wherever the
    # real structural break actually is as training data grows, rather than
    # assuming there will always be exactly two standout points the way
    # today's 31 samples happen to show.
    sub_scores = training_scores[:-1]
    gaps = [sub_scores[i + 1] - sub_scores[i] for i in range(len(sub_scores) - 1)]
    gap_index = int(np.argmax(gaps))
    warning_threshold = (sub_scores[gap_index] + sub_scores[gap_index + 1]) / 2
    blocking_threshold = training_max * BLOCKING_MARGIN
    if mahalanobis > blocking_threshold:
        severity = "blocking"
    elif mahalanobis > warning_threshold:
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
            "blocking_threshold": blocking_threshold,
            "warning_threshold": warning_threshold,
        },
    )


def run_anomaly_checks(request: ValidateRequest) -> list[Anomaly]:
    return [check_ood(request)]
