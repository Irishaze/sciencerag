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
# Reasonableness band for scalar outputs once the benchmark case's geometry
# is confirmed to match: repeated solves of the same geometry/BCs should
# reproduce closely, so this is deliberately tight, not a physical-model
# tolerance.
BENCHMARK_SCALAR_RELATIVE_TOLERANCE = 0.05


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
        band = BENCHMARK_SCALAR_RELATIVE_TOLERANCE * max(abs(reference), 1e-9)
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
