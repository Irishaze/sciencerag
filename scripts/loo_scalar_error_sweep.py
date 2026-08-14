"""Real leave-one-out CV sweep over the 31 known COMSOL benchmark samples,
using the currently-deployed latent surrogate's own training config
(rbf_svr, see tec_bridge.load_latent_model()['regressor_config']).

Produces, per scalar field, the real held-out relative-error distribution —
used to calibrate sciencerag.validate's per-field benchmark tolerance
(evaluation.py's BENCHMARK_SCALAR_RELATIVE_TOLERANCE_BY_FIELD) and KG
candidate confidence (kg_candidates.py's confidence formula), replacing
constants that were previously either a single flat 5% tolerance or fixed
0.7/0.4 verdict buckets with no real-data backing (2026-08-14/15 finding:
the flat tolerance is badly mismatched per field — delta_T_max_K/
figure_of_merit_1_per_K generalize to <2% error, while total_resistance_ohm/
optimal_current_A/max_heat_dissipation_W have 25-55%+ median leave-one-out
error against this same 31-sample dataset).

The _fit_pipeline/_make_regressor/_predict_vector logic below is copied
verbatim from tec_surrogate/scripts/09_train_from_reports.py (not
reimplemented/guessed) — that module has a top-level `import matplotlib`
unrelated to training/prediction that isn't installed in sciencerag's venv,
so importing it directly fails; this avoids adding a new dependency just to
reuse three small, already-correct sklearn wrapper functions.

    uv run python scripts/loo_scalar_error_sweep.py
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sciencerag.validate import tec_bridge


def _latent_dimension(pca: PCA, threshold: float) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def _make_regressor(config: dict):
    if config["kind"] == "ridge":
        return Ridge(alpha=config["alpha"])
    if config["kind"] == "rbf_svr":
        return MultiOutputRegressor(SVR(C=config["C"], gamma=config["gamma"], epsilon=config["epsilon"]))
    raise ValueError(f"Unknown regressor kind: {config['kind']}")


def _fit_pipeline(X: np.ndarray, Y: np.ndarray, variance_threshold: float, regressor_config: dict) -> dict:
    input_scaler = StandardScaler().fit(X)
    output_scaler = StandardScaler().fit(Y)
    Y_scaled = output_scaler.transform(Y)
    pca_probe = PCA().fit(Y_scaled)
    latent_dim = _latent_dimension(pca_probe, variance_threshold)
    pca = PCA(n_components=latent_dim).fit(Y_scaled)
    latent = pca.transform(Y_scaled)
    regressor = _make_regressor(regressor_config).fit(input_scaler.transform(X), latent)
    return {"input_scaler": input_scaler, "output_scaler": output_scaler, "pca": pca, "regressor": regressor}


def _predict_vector(pipeline: dict, X: np.ndarray) -> np.ndarray:
    latent = pipeline["regressor"].predict(pipeline["input_scaler"].transform(X))
    if latent.ndim == 1:
        latent = latent[:, None]
    scaled_output = pipeline["pca"].inverse_transform(latent)
    return pipeline["output_scaler"].inverse_transform(scaled_output)


def run_sweep() -> dict:
    model = tec_bridge.load_latent_model()
    dataset = tec_bridge.load_report_dataset()

    X = dataset["X"]
    scalar_outputs = dataset["scalar_outputs"]
    cop_surfaces = dataset["cop_surfaces"]
    scalar_names = list(dataset["scalar_names"])
    n = len(X)
    Y = np.concatenate([scalar_outputs, cop_surfaces.reshape(n, -1)], axis=1)

    variance_threshold = model["variance_threshold"]
    regressor_config = model["regressor_config"]

    per_field: dict[str, list[float]] = {name: [] for name in scalar_names}
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        pipeline = _fit_pipeline(X[train_idx], Y[train_idx], variance_threshold, regressor_config)
        pred = _predict_vector(pipeline, X[i : i + 1])[0]
        true = Y[i]
        for k, name in enumerate(scalar_names):
            ref = true[k]
            if abs(ref) < 1e-12:
                continue
            per_field[name].append(abs(pred[k] - ref) / abs(ref))

    summary = {}
    for name, vals in per_field.items():
        arr = np.array(vals)
        summary[name] = {
            "n": len(arr),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
        }
    return {"regressor_config": regressor_config, "n_samples": n, "per_field": summary}


if __name__ == "__main__":
    result = run_sweep()
    print(json.dumps(result, indent=2))
