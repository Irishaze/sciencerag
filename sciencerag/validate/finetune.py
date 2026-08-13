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
        # "error" in evidence means the check couldn't be scored at all
        # (e.g. checks.py's dimension-mismatch path) — confirmed live that
        # this branch previously fired anyway, claiming "OOD score in the
        # upper tail... near this design point" for a run with no computed
        # OOD score and no valid latent coordinate to expand coverage near.
        # Only claim a coverage-expansion direction when a real score backs
        # it up.
        if anomaly.check in _HYPERPARAMETER_DIRECTION_BY_CHECK and "error" not in anomaly.evidence:
            directions.append(_HYPERPARAMETER_DIRECTION_BY_CHECK[anomaly.check])

    if not samples:
        return None

    return SurrogateUpdateSuggestion(
        recommended_training_samples=samples,
        hyperparameter_direction="; ".join(directions) if directions else (
            "no specific hyperparameter direction — samples came from "
            "benchmark/prior deviation and/or a physics check that could not "
            "be scored, not a real computed OOD/residual signal"
        ),
    )
