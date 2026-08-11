"""4.3 model fine-tuning suggestions (spec §4.3).

Proposal only — nothing here trains anything. Selects samples off two
signals per spec: error-driven (evaluation deviations against benchmark/
prior) and uncertainty-driven (warning-severity anomalies — OOD or elevated
residuals; blocking-severity anomalies never reach this function, since the
router short-circuits 4.3/4.4 for blocked runs per spec §4.1).

A run that triggers neither signal returns None — an unremarkable,
in-distribution, in-benchmark run has nothing actionable to suggest, and a
suggestion produced anyway would just be training-set noise.
"""

from __future__ import annotations

from sciencerag.validate.models import (
    Anomaly,
    Evaluation,
    RecommendedSample,
    SurrogateUpdateSuggestion,
    ValidateRequest,
)

# Heuristic reweighting nudge per triggering check — not derived from any
# calibration data, just a documented direction-of-travel a human reviewer
# can accept, adjust, or reject before any retraining actually happens.
# Empty for now: ood is the only check left (energy_balance/pde_residual
# were removed, see checks.py) and it isn't a loss term to reweight, just a
# training-coverage signal (see _HYPERPARAMETER_DIRECTION_BY_CHECK below).
_LOSS_REWEIGHT_BY_CHECK: dict[str, float] = {}

_HYPERPARAMETER_DIRECTION_BY_CHECK = {
    "ood": (
        "OOD score in the upper tail of the training self-distance "
        "distribution — consider expanding Sobol/training coverage near "
        "this design point"
    ),
}


def suggest_surrogate_update(
    request: ValidateRequest, anomalies: list[Anomaly], evaluation: Evaluation
) -> SurrogateUpdateSuggestion | None:
    samples: list[RecommendedSample] = []
    loss_reweighting: dict[str, float] = {}
    directions: list[str] = []

    for deviation in evaluation.deviations:
        if deviation.verdict != "deviation":
            continue
        samples.append(
            RecommendedSample(
                run_id=request.run_id,
                region=f"{deviation.source}:{deviation.field}",
                reason=(
                    f"actual={deviation.actual} outside reference range "
                    f"[{deviation.reference_min}, {deviation.reference_max}] "
                    f"(reference={deviation.reference_id})"
                ),
            )
        )

    for anomaly in anomalies:
        if anomaly.severity != "warning":
            continue
        samples.append(
            RecommendedSample(
                run_id=request.run_id,
                region=f"check:{anomaly.check}",
                reason=f"{anomaly.check} flagged warning: {anomaly.evidence}",
            )
        )
        if anomaly.check in _LOSS_REWEIGHT_BY_CHECK:
            loss_reweighting[anomaly.check] = _LOSS_REWEIGHT_BY_CHECK[anomaly.check]
        if anomaly.check in _HYPERPARAMETER_DIRECTION_BY_CHECK:
            directions.append(_HYPERPARAMETER_DIRECTION_BY_CHECK[anomaly.check])

    if not samples:
        return None

    return SurrogateUpdateSuggestion(
        recommended_training_samples=samples,
        loss_reweighting=loss_reweighting,
        hyperparameter_direction="; ".join(directions) if directions else (
            "no specific hyperparameter direction — signal came from "
            "benchmark/prior deviation only, not a physics-check warning"
        ),
    )
