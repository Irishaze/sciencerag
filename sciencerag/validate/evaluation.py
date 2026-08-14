"""4.2 result evaluation (spec §4.2): benchmark comparison + prior
comparison, merged into one Evaluation.

v1 scope, matching the spec's own precedent for first-iteration
simplification (§3.6 "v1 先给出...简化归因"):
- 4.2.1 benchmark comparison only fires when this run's design_parameters
  match a benchmark case almost exactly (same geometry) — comparing scalar
  outputs across *different* geometries wouldn't quantify a deviation, it
  would just restate that different designs perform differently. Everything
  else reports insufficient_benchmark honestly instead of a fabricated
  comparison.
- 4.2.2 prior comparison only handles `kind="parameter_range"` priors
  against `design_parameters`; scaling_relationship/candidate_config/caution
  aren't scalar range checks and are left for a later iteration.
"""

from __future__ import annotations

from sciencerag.priors.contract import GEOMETRY_FREE_PARAMS
from sciencerag.validate import tec_bridge
from sciencerag.validate.models import Deviation, Evaluation, ValidateRequest

_CONTRACT_UNIT_BY_NAME = {param["name"]: param["unit"] for param in GEOMETRY_FREE_PARAMS}

# Relative tolerance for treating a benchmark case as "the same geometry" as
# this run (not just a nearby one) — tight, because different designs are
# expected to perform differently, so only a near-exact geometry match makes
# a scalar comparison meaningful (see module docstring).
GEOMETRY_MATCH_RELATIVE_TOLERANCE = 0.01
# Fallback for any scalar field not in BENCHMARK_SCALAR_RELATIVE_TOLERANCE_BY_FIELD
# below (e.g. a new field added to the contract before the sweep is re-run).
BENCHMARK_SCALAR_RELATIVE_TOLERANCE = 0.05

# A single flat 5% band applied to every scalar field was checked against
# real data (2026-08-14/15, scripts/loo_scalar_error_sweep.py — real
# leave-one-out cross-validation of the deployed latent surrogate over its
# own 31-sample training set, not synthetic/assumed error) and found badly
# mismatched: delta_T_max_K/figure_of_merit_1_per_K generalize tightly
# (<1.5% p90 relative error — 5% was already generous for these), while
# total_resistance_ohm/optimal_current_A/max_heat_dissipation_W have 90th-
# percentile leave-one-out error of 89-219% — meaning a flat 5% band is
# essentially unreachable for those fields from a genuine new prediction
# (confirmed separately: every historical "consistent" verdict in
# logs/audit.jsonl for benchmark_comparison had exactly 0.0000 deviation,
# i.e. the request had echoed a benchmark value back verbatim rather than
# supplying a real surrogate prediction — a flat 5% tolerance was silently
# never being cleared by real predictions on the wide-error fields, so
# "consistent" could only ever fire on exact echoes).
#
# Per-field values here are min(p90 leave-one-out relative error, 0.05),
# capped at 0.5 (50%) rather than used raw — the sweep's own p90 for
# total_resistance_ohm is 219%, and widening tolerance that far would make
# the check accept almost anything, defeating its purpose. Capping at 0.5
# keeps the check meaningful while still reflecting that this field's real
# reliability is much worse than delta_T_max_K's — deviation_found will
# likely still fire often for total_resistance_ohm/optimal_current_A even
# with the widened band, which is an honest signal about surrogate quality
# for those two fields specifically, not a bug in this tolerance table.
BENCHMARK_SCALAR_RELATIVE_TOLERANCE_BY_FIELD = {
    "delta_T_max_K": 0.05,
    "figure_of_merit_1_per_K": 0.05,
    "optimal_voltage_V": 0.30,
    "max_heat_dissipation_W": 0.5,
    "optimal_current_A": 0.5,
    "total_resistance_ohm": 0.5,
}


def _find_matching_benchmark_case(design_parameters: dict[str, float]) -> int | None:
    dataset = tec_bridge.load_report_dataset()
    input_names = list(dataset["input_names"])
    X = dataset["X"]
    columns: dict[str, int] = {}
    for contract_name, latent_name in tec_bridge.CONTRACT_TO_LATENT_INPUT.items():
        if contract_name in design_parameters and latent_name in input_names:
            columns[contract_name] = input_names.index(latent_name)
    if not columns:
        return None
    for row_index in range(len(X)):
        matched = True
        for contract_name, column_index in columns.items():
            actual = design_parameters[contract_name]
            reference = float(X[row_index, column_index])
            tolerance = GEOMETRY_MATCH_RELATIVE_TOLERANCE * max(abs(reference), 1e-9)
            if abs(actual - reference) > tolerance:
                matched = False
                break
        if matched:
            return row_index
    return None


def _benchmark_deviations(request: ValidateRequest) -> tuple[list[Deviation], bool]:
    """Returns (deviations, benchmark_available)."""
    if not request.scalar_results or not request.design_parameters:
        return [], False
    case_index = _find_matching_benchmark_case(request.design_parameters)
    if case_index is None:
        return [], False
    dataset = tec_bridge.load_report_dataset()
    scalar_names = list(dataset["scalar_names"])
    filenames = list(dataset["filenames"])
    reference_id = str(filenames[case_index])
    deviations = []
    for field, actual in request.scalar_results.items():
        if field not in scalar_names:
            continue
        reference = float(dataset["scalar_outputs"][case_index, scalar_names.index(field)])
        tolerance = BENCHMARK_SCALAR_RELATIVE_TOLERANCE_BY_FIELD.get(
            field, BENCHMARK_SCALAR_RELATIVE_TOLERANCE
        )
        band = tolerance * max(abs(reference), 1e-9)
        verdict = "within_range" if abs(actual - reference) <= band else "deviation"
        deviations.append(
            Deviation(
                field=field,
                source="benchmark_comparison",
                reference_id=reference_id,
                actual=actual,
                reference_min=reference - band,
                reference_max=reference + band,
                verdict=verdict,
            )
        )
    return deviations, True


def _prior_deviations(request: ValidateRequest) -> list[Deviation]:
    deviations = []
    for prior in request.priors:
        if prior.kind != "parameter_range" or prior.field is None:
            continue
        if prior.field not in request.design_parameters:
            continue
        value = prior.value
        # ParameterRangeValue by construction here (Prior's own validator
        # ties kind="parameter_range" to this value type).
        if value.min is None and value.max is None:
            continue  # only a `typical` point given — no range to check against
        contract_unit = _CONTRACT_UNIT_BY_NAME.get(prior.field)
        if contract_unit is not None and value.unit != contract_unit:
            # spec §3.8: v1 does no unit conversion — comparing under a unit
            # mismatch would silently produce a wrong verdict, so skip
            # rather than guess.
            continue
        actual = request.design_parameters[prior.field]
        below = value.min is not None and actual < value.min
        above = value.max is not None and actual > value.max
        verdict = "deviation" if (below or above) else "within_range"
        deviations.append(
            Deviation(
                field=prior.field,
                source="prior_comparison",
                reference_id=prior.prior_id,
                actual=actual,
                reference_min=value.min,
                reference_max=value.max,
                verdict=verdict,
            )
        )
    return deviations


def evaluate(request: ValidateRequest) -> Evaluation:
    benchmark_deviations, benchmark_available = _benchmark_deviations(request)
    prior_deviations = _prior_deviations(request)
    all_deviations = benchmark_deviations + prior_deviations
    sources = [source for prior in request.priors for source in prior.sources]

    if any(deviation.verdict == "deviation" for deviation in all_deviations):
        verdict = "deviation_found"
    elif not benchmark_available and not prior_deviations:
        verdict = "insufficient_benchmark"
    else:
        verdict = "consistent"

    return Evaluation(verdict=verdict, deviations=all_deviations, sources=sources)
