"""Adapter into tec_surrogate's trained artifacts and physics_foundation
package.

Isolated here so the rest of sciencerag.validate never touches
tec_surrogate's sys.path/import mechanics or on-disk checkpoint layout
directly. tec_surrogate is a sibling directory, not an installed package
(same pattern its own scripts/*.py use: sys.path.insert then `import
physics_foundation`).

Known coverage gap (M2 planning, mentor 2026-08-03): the trained latent
surrogate and both field checkpoints were fit on an 8-geometry-parameter
model with no heatsink — 4 of the contract's 12 geometry_free names
(sink_base_h/sink_fin_h/sink_fin_w/sink_fin_n) have no trained encoder
behind them yet. SUPPORTED_GEOMETRY_NAMES reflects that; callers asking
about the other 4 get an explicit unsupported-name error, not a silent
wrong answer.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

TEC_SURROGATE_ROOT = Path(__file__).resolve().parents[2] / "tec_surrogate"
if str(TEC_SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEC_SURROGATE_ROOT))

from physics_foundation import (  # noqa: E402
    ComponentGraph,
    ComponentGraphSepONet,
    InterfaceSamples,
    LossScales,
    compose_multi_pair_graph,
    interface_conservation_loss,
    load_component_case,
    load_component_interfaces,
    steady_thermoelectric_residual_loss,
)

LATENT_MODEL_PATH = TEC_SURROGATE_ROOT / "data" / "models" / "comsol_latent_surrogate.joblib"
DATASET_PATH = TEC_SURROGATE_ROOT / "data" / "processed" / "comsol_report_dataset.npz"
ONE_PAIR_CASE_PATH = TEC_SURROGATE_ROOT / "data" / "component_cases" / "tec_1pair_dset3.npz"
REAL_CHECKPOINT_PATH = TEC_SURROGATE_ROOT / "outputs" / "component_graph_seponet_real.pt"
HIERARCHICAL_CHECKPOINT_PATH = TEC_SURROGATE_ROOT / "outputs" / "component_graph_seponet_20pairs.pt"

# The latent surrogate's INPUT_NAMES (scripts/09_train_from_reports.py) use
# unit-suffixed names in the same units as the sim_params.json contract
# (spec §3.6) — this is the name+unit translation table between the two.
# "n_pairs"/"Tref_K" are contract `derived`/`operating_condition` fields,
# not `geometry_free`, so they aren't in SUPPORTED_GEOMETRY_NAMES.
CONTRACT_TO_LATENT_INPUT = {
    "length": "length_mm",
    "width": "width_mm",
    "height": "height_mm",
    "d_conductor": "d_conductor_um",
    "d_ceramics": "d_ceramics_mm",
    "leg_length": "leg_length_mm",
    "leg_width": "leg_width_mm",
    "pitch": "pitch_mm",
}
SUPPORTED_GEOMETRY_NAMES = frozenset(CONTRACT_TO_LATENT_INPUT)
UNSUPPORTED_GEOMETRY_NAMES = frozenset(
    {"sink_base_h", "sink_fin_h", "sink_fin_w", "sink_fin_n"}
)

SCALAR_NAMES = (
    "delta_T_max_K",
    "optimal_current_A",
    "optimal_voltage_V",
    "total_resistance_ohm",
    "max_heat_dissipation_W",
    "figure_of_merit_1_per_K",
)

SCALAR_UNITS = {
    "delta_T_max_K": "K",
    "optimal_current_A": "A",
    "optimal_voltage_V": "V",
    "total_resistance_ohm": "ohm",
    "max_heat_dissipation_W": "W",
    "figure_of_merit_1_per_K": "1/K",
}


@lru_cache(maxsize=1)
def load_latent_model() -> dict:
    return joblib.load(LATENT_MODEL_PATH)


@lru_cache(maxsize=1)
def load_report_dataset() -> dict[str, np.ndarray]:
    """The 31 unique COMSOL-report training samples backing the latent
    surrogate — used both for the OOD reference distribution and, here, as
    the (small) 4.2.1 benchmark database of known solved cases."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


@lru_cache(maxsize=1)
def load_training_latent() -> np.ndarray:
    """True PCA scores (z) of the known training reports — the historical
    distribution 4.1.3 OOD detection scores a new run's z against.

    Deliberately the *true* PCA transform of each training report's actual
    output vector, not the regressor's input->z prediction: a run being
    validated already has a real simulated result by the time validate is
    called (spec §4: called after 04 sim + 05 normalization), so encoding
    should reflect what actually happened, not a pre-simulation guess.
    """
    model = load_latent_model()
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        Y = data["Y"]
    scaled = model["output_scaler"].transform(Y)
    return model["pca"].transform(scaled)


@lru_cache(maxsize=1)
def load_real_checkpoint() -> tuple[ComponentGraphSepONet, dict]:
    """The single-geometry checkpoint trained on real tec_1pair_dset3.npz
    data (scripts/13_train_real_component_graph.py). Only source of a
    genuine, COMSOL-anchored conservation check (4.1.1)."""
    checkpoint = torch.load(REAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = ComponentGraphSepONet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@lru_cache(maxsize=1)
def load_hierarchical_checkpoint() -> tuple[ComponentGraphSepONet, dict]:
    """The 1-20 pair composed checkpoint (scripts/14_train_multipair_component_graph.py).
    Used for 4.1.2 PDE residual only, on the model's raw decode() output —
    field_service.py's UI-facing predict() applies post-hoc calibration
    snapping (hard temperature/voltage constraints) that would make a
    residual computed on its output meaningless."""
    checkpoint = torch.load(
        HIERARCHICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model = ComponentGraphSepONet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _normalise_globals(
    graph: ComponentGraph, mean: torch.Tensor, scale: torch.Tensor
) -> ComponentGraph:
    return replace(graph, global_features=(graph.global_features - mean) / scale)


def load_real_one_pair_graph(solution_index: int) -> ComponentGraph:
    """A solved one-pair case, normalized exactly as at training time."""
    _model, checkpoint = load_real_checkpoint()
    graph = load_component_case(ONE_PAIR_CASE_PATH, solution_index)
    return _normalise_globals(graph, checkpoint["global_mean"], checkpoint["global_scale"])


@lru_cache(maxsize=1)
def load_one_pair_interfaces() -> InterfaceSamples:
    return load_component_interfaces(ONE_PAIR_CASE_PATH)


def compose_hierarchical_graph(
    solution_index: int, n_pairs: int
) -> ComponentGraph:
    """Reuses the same base-graph + compose step field_service.py's
    production predict() path takes (self.base_graphs[nearest_solution]),
    minus the post-hoc calibration snapping — see load_hierarchical_checkpoint."""
    base_graph = load_component_case(ONE_PAIR_CASE_PATH, solution_index)
    current = float(base_graph.global_features[0, 0])
    graph, _layout = compose_multi_pair_graph(
        base_graph, n_pairs, current=current, max_probes_per_component=48
    )
    return graph


DEFAULT_LOSS_SCALES = LossScales(
    temperature=50.0, potential=1.0, heat_flux=1.0e6, current_density=1.0e6
)


def conservation_residual(solution_index: int) -> dict[str, float]:
    """4.1.1: interface conservation residual terms for a solved one-pair
    case, using the real COMSOL-anchored checkpoint. Only defined for
    n_pairs=1 — compose_multi_pair_graph doesn't produce matched interface
    samples for the module-to-module boundaries it invents, so there is no
    real interface data to check conservation against for n_pairs > 1."""
    model, _checkpoint = load_real_checkpoint()
    graph = load_real_one_pair_graph(solution_index)
    interfaces = load_one_pair_interfaces()
    node_state = model.encode_nodes(graph)
    _total, terms = interface_conservation_loss(
        model, graph, node_state, interfaces, DEFAULT_LOSS_SCALES
    )
    return {name: float(value.detach()) for name, value in terms.items()}


def pde_residual(solution_index: int, n_pairs: int) -> dict[str, float]:
    """4.1.2: steady thermoelectric PDE residual, evaluated at the graph's
    own probe points. Works for any n_pairs via the hierarchical checkpoint;
    n_pairs=1 uses the same checkpoint (not the real-only one) so the
    residual magnitude is comparable across n_pairs on one consistent
    model."""
    model, _checkpoint = load_hierarchical_checkpoint()
    graph = compose_hierarchical_graph(solution_index, n_pairs)
    node_state = model.encode_nodes(graph)
    _total, terms = steady_thermoelectric_residual_loss(
        model, graph, node_state, graph.probe_components, graph.probe_coords, DEFAULT_LOSS_SCALES
    )
    return {name: float(value.detach()) for name, value in terms.items()}


# Neither loss's absolute scale is calibrated against real physical
# tolerances (tec_surrogate/docs/PHYSICS_FOUNDATION.md lists "calibrated
# loss scales from the COMSOL dataset" under "Not implemented yet") — an
# absolute threshold on these numbers would be fabricated. Instead, severity
# is relative: how does this run's residual compare against the same
# residual computed on the 11 known-solved one-pair operating points, which
# are real solved COMSOL data and therefore a legitimate "in-distribution"
# baseline even though their absolute scale is uncalibrated.
ALL_SOLUTION_INDICES = tuple(range(11))


def conservation_residual_total(solution_index: int) -> float:
    return sum(conservation_residual(solution_index).values())


def pde_residual_total(solution_index: int, n_pairs: int) -> float:
    return sum(pde_residual(solution_index, n_pairs).values())


@lru_cache(maxsize=1)
def conservation_baseline() -> tuple[float, ...]:
    return tuple(conservation_residual_total(index) for index in ALL_SOLUTION_INDICES)


@lru_cache(maxsize=20)
def pde_residual_baseline(n_pairs: int) -> tuple[float, ...]:
    return tuple(pde_residual_total(index, n_pairs) for index in ALL_SOLUTION_INDICES)
