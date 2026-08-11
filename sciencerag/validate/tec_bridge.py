"""Adapter into tec_surrogate's trained artifacts.

Isolated here so the rest of sciencerag.validate never touches
tec_surrogate's sys.path/import mechanics or on-disk checkpoint layout
directly. tec_surrogate is a sibling directory, not an installed package
(same pattern its own scripts/*.py use: sys.path.insert then `import
physics_foundation`).

Only the latent surrogate (comsol_latent_surrogate.joblib) is loaded here
now — the component-graph field checkpoints (component_graph_seponet_real.pt
/ _20pairs.pt) and everything in physics_foundation that composed/decoded
them were removed along with checks.py's energy_balance/pde_residual: both
depended on a compositional multi-pair model that was never a substitute
for real multi-pair COMSOL calibration data (its own training script's
docstring said so), trained/validated only up to 20 pairs, with no
principled way to answer for a real commercial module (127+ pairs) without
silently extrapolating.

Known coverage gap (M2 planning, mentor 2026-08-03): the trained latent
surrogate was fit on an 8-geometry-parameter model with no heatsink — 4 of
the contract's 12 geometry_free names (sink_base_h/sink_fin_h/sink_fin_w/
sink_fin_n) have no trained encoder behind them yet. SUPPORTED_GEOMETRY_NAMES
reflects that; callers asking about the other 4 get an explicit
unsupported-name error, not a silent wrong answer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np

TEC_SURROGATE_ROOT = Path(__file__).resolve().parents[2] / "tec_surrogate"
if str(TEC_SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEC_SURROGATE_ROOT))

LATENT_MODEL_PATH = TEC_SURROGATE_ROOT / "data" / "models" / "comsol_latent_surrogate.joblib"
DATASET_PATH = TEC_SURROGATE_ROOT / "data" / "processed" / "comsol_report_dataset.npz"

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
