"""Train a latent-space surrogate from COMSOL DOCX reports.

The report output vector contains six scalar performance metrics followed by
the COP surface sampled over the common (dT0, I0) grid. PCA compresses this
vector, and ridge regression maps the ten design inputs to the latent scores.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
from docx import Document
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FIGURES, MODELS_DIR, OUTPUTS, PROCESSED, RAW


INPUT_SPECS = [
    ("length_mm", "length", "mm"),
    ("width_mm", "width", "mm"),
    ("height_mm", "height", "mm"),
    ("d_conductor_um", "d_conductor", "um"),
    ("d_ceramics_mm", "d_ceramics", "mm"),
    ("leg_length_mm", "leg_length", "mm"),
    ("leg_width_mm", "leg_width", "mm"),
    ("pitch_mm", "pitch", "mm"),
    ("n_pairs", "N", "1"),
    ("Tref_K", "Tref", "K"),
]
INPUT_NAMES = [spec[0] for spec in INPUT_SPECS]

SCALAR_HEADERS = {
    "最大温差(K)": "delta_T_max_K",
    "最大温差所需电流(A)": "optimal_current_A",
    "最大温差所需电压(V)": "optimal_voltage_V",
    "总电阻(Ω)": "total_resistance_ohm",
    "最大热耗散(W)": "max_heat_dissipation_W",
    "品质因子(1/K)": "figure_of_merit_1_per_K",
}
SCALAR_NAMES = list(SCALAR_HEADERS.values())

DEFAULT_REPORT_DIR = RAW / "doc"
DEFAULT_MODEL_PATH = MODELS_DIR / "comsol_latent_surrogate.joblib"
DEFAULT_DATASET_PATH = PROCESSED / "comsol_report_dataset.npz"
DEFAULT_DATASET_JSON = PROCESSED / "comsol_report_dataset.json"
DEFAULT_SUMMARY_PATH = OUTPUTS / "comsol_latent_training.json"
DEFAULT_FIGURE_PATH = FIGURES / "comsol_latent_training.png"

NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
DT_HEADER_RE = re.compile(r"dT0\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*K", re.I)


def parse_comsol_number(text: str) -> float:
    """Parse a number including COMSOL's Unicode scientific notation."""
    normalized = (
        str(text)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("Ｅ", "E")
        .replace("ｅ", "e")
        .replace("\xa0", " ")
    )
    match = NUMBER_RE.search(normalized)
    if not match:
        raise ValueError(f"No numeric value in {text!r}")
    return float(match.group(0))


def _normalized_unit(text: str) -> str | None:
    match = re.search(r"\[\s*([^\]]+)\s*\]", str(text))
    if not match:
        return None
    return match.group(1).strip().replace("µ", "u").replace("μ", "u")


def parse_quantity(text: str, target_unit: str) -> float:
    value = parse_comsol_number(text)
    if target_unit == "1":
        return value

    source_unit = _normalized_unit(text)
    if source_unit is None:
        raise ValueError(f"Missing unit in {text!r}")
    if source_unit == target_unit:
        return value

    to_si = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "K": 1.0}
    if source_unit not in to_si or target_unit not in to_si:
        raise ValueError(f"Cannot convert {source_unit!r} to {target_unit!r}")
    return value * to_si[source_unit] / to_si[target_unit]


def _normalize_header(text: str) -> str:
    return re.sub(r"\s+", "", str(text).replace("（", "(").replace("）", ")"))

def _table_rows(table: Any) -> list[list[str]]:
    return [[cell.text.strip() for cell in row.cells] for row in table.rows]


def parse_report(path: Path) -> dict[str, Any]:
    """Parse one report by semantic field names instead of table positions."""
    document = Document(path)
    parameter_rows: dict[str, list[str]] = {}
    scalars: dict[str, float] = {}
    cop_values: dict[tuple[float, float], float] = {}

    for table in document.tables:
        rows = _table_rows(table)
        if not rows or not rows[0]:
            continue

        header = rows[0]
        if len(header) >= 3 and header[:3] == ["名称", "表达式", "值"]:
            for row in rows[1:]:
                if len(row) >= 3 and row[0]:
                    previous = parameter_rows.get(row[0])
                    if previous is not None and previous[1:3] != row[1:3]:
                        raise ValueError(f"Conflicting parameter {row[0]!r}")
                    parameter_rows[row[0]] = row
            continue

        normalized_header = _normalize_header(header[0])
        if len(rows) == 2 and len(header) == 1 and normalized_header in SCALAR_HEADERS:
            scalars[SCALAR_HEADERS[normalized_header]] = parse_comsol_number(rows[1][0])
            continue

        if header[0] == "I0" and any("性能系数" in cell for cell in header[1:]):
            dts: list[float] = []
            for column_header in header[1:]:
                match = DT_HEADER_RE.search(column_header)
                if not match:
                    raise ValueError(f"Cannot parse COP column {column_header!r}")
                dts.append(float(match.group(1)))

            for row in rows[1:]:
                if not row or not row[0]:
                    continue
                current = parse_comsol_number(row[0])
                for column, dt in enumerate(dts, start=1):
                    value = parse_comsol_number(row[column])
                    key = (current, dt)
                    if key in cop_values and not np.isclose(cop_values[key], value, rtol=1e-8, atol=1e-10):
                        raise ValueError(f"Conflicting duplicate COP point I0={current}, dT0={dt}")
                    cop_values[key] = value

    missing_parameters = [key for _, key, _ in INPUT_SPECS if key not in parameter_rows]
    missing_scalars = [name for name in SCALAR_NAMES if name not in scalars]
    if missing_parameters:
        raise ValueError(f"Missing parameters: {missing_parameters}")
    if missing_scalars:
        raise ValueError(f"Missing scalar results: {missing_scalars}")
    if not cop_values:
        raise ValueError("Missing COP surface")

    inputs: dict[str, float] = {}
    for output_name, report_name, target_unit in INPUT_SPECS:
        row = parameter_rows[report_name]
        source_text = row[2] if report_name == "N" else row[1]
        inputs[output_name] = parse_quantity(source_text, target_unit)

    currents = sorted({key[0] for key in cop_values})
    delta_t_values = sorted({key[1] for key in cop_values})
    expected_points = {(current, dt) for dt in delta_t_values for current in currents}
    missing_points = sorted(expected_points - set(cop_values))
    if missing_points:
        raise ValueError(f"Incomplete COP grid: missing {missing_points}")

    cop_surface = np.array(
        [[cop_values[(current, dt)] for current in currents] for dt in delta_t_values],
        dtype=float,
    )
    return {
        "file": path.name,
        "inputs": inputs,
        "scalars": scalars,
        "currents": currents,
        "delta_t_values": delta_t_values,
        "cop_surface": cop_surface,
    }


def extract_dataset(report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, Any]:
    """Parse, validate, and deduplicate all DOCX reports."""
    paths = sorted(Path(report_dir).glob("*.docx"), key=lambda path: path.name.lower())
    if not paths:
        raise FileNotFoundError(f"No DOCX reports found in {report_dir}")

    parsed: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            parsed.append(parse_report(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise RuntimeError("Report parsing failed:\n  " + "\n  ".join(errors))

    reference_currents = parsed[0]["currents"]
    reference_dts = parsed[0]["delta_t_values"]
    for sample in parsed[1:]:
        if sample["currents"] != reference_currents or sample["delta_t_values"] != reference_dts:
            raise ValueError(f"{sample['file']} uses a different COP grid")

    unique_samples: list[dict[str, Any]] = []
    duplicate_files: list[dict[str, str]] = []
    seen: dict[tuple[float, ...], dict[str, Any]] = {}
    for sample in parsed:
        x = np.array([sample["inputs"][name] for name in INPUT_NAMES], dtype=float)
        scalar = np.array([sample["scalars"][name] for name in SCALAR_NAMES], dtype=float)
        y = np.concatenate([scalar, sample["cop_surface"].reshape(-1)])
        key = tuple(np.round(x, decimals=10))
        if key in seen:
            original = seen[key]
            original_y = np.concatenate(
                [
                    np.array([original["scalars"][name] for name in SCALAR_NAMES]),
                    original["cop_surface"].reshape(-1),
                ]
            )
            if not np.allclose(y, original_y, rtol=1e-7, atol=1e-9):
                raise ValueError(
                    f"Duplicate inputs have conflicting results: {original['file']} and {sample['file']}"
                )
            duplicate_files.append({"file": sample["file"], "duplicate_of": original["file"]})
            continue
        seen[key] = sample
        unique_samples.append(sample)

    X = np.array(
        [[sample["inputs"][name] for name in INPUT_NAMES] for sample in unique_samples], dtype=float
    )
    scalar_outputs = np.array(
        [[sample["scalars"][name] for name in SCALAR_NAMES] for sample in unique_samples], dtype=float
    )
    cop_surfaces = np.array([sample["cop_surface"] for sample in unique_samples], dtype=float)
    Y = np.concatenate([scalar_outputs, cop_surfaces.reshape(len(unique_samples), -1)], axis=1)
    cop_names = [
        f"COP_dT{dt:g}K_I0_{current:g}"
        for dt in reference_dts
        for current in reference_currents
    ]

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Dataset contains non-finite values")
    return {
        "X": X,
        "Y": Y,
        "scalar_outputs": scalar_outputs,
        "cop_surfaces": cop_surfaces,
        "filenames": [sample["file"] for sample in unique_samples],
        "input_names": INPUT_NAMES,
        "scalar_names": SCALAR_NAMES,
        "output_names": SCALAR_NAMES + cop_names,
        "currents": reference_currents,
        "delta_t_values": reference_dts,
        "duplicates": duplicate_files,
        "source_file_count": len(paths),
    }


def _latent_dimension(pca: PCA, threshold: float) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumulative, threshold) + 1)


def _fit_pipeline(
    X: np.ndarray,
    Y: np.ndarray,
    variance_threshold: float,
    regressor_config: dict[str, Any],
) -> dict[str, Any]:
    input_scaler = StandardScaler().fit(X)
    output_scaler = StandardScaler().fit(Y)
    Y_scaled = output_scaler.transform(Y)
    pca_probe = PCA().fit(Y_scaled)
    latent_dim = _latent_dimension(pca_probe, variance_threshold)
    pca = PCA(n_components=latent_dim).fit(Y_scaled)
    latent = pca.transform(Y_scaled)
    regressor = _make_regressor(regressor_config).fit(input_scaler.transform(X), latent)
    return {
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "pca": pca,
        "regressor": regressor,
        "latent": latent,
        "latent_dim": latent_dim,
        "all_explained_variance_ratio": pca_probe.explained_variance_ratio_,
    }


def _make_regressor(config: dict[str, Any]) -> Any:
    if config["kind"] == "ridge":
        return Ridge(alpha=config["alpha"])
    if config["kind"] == "rbf_svr":
        return MultiOutputRegressor(
            SVR(C=config["C"], gamma=config["gamma"], epsilon=config["epsilon"])
        )
    raise ValueError(f"Unknown regressor kind: {config['kind']}")


def _predict_vector(pipeline: dict[str, Any], X: np.ndarray) -> np.ndarray:
    latent = pipeline["regressor"].predict(pipeline["input_scaler"].transform(X))
    if latent.ndim == 1:
        latent = latent[:, None]
    scaled_output = pipeline["pca"].inverse_transform(latent)
    return pipeline["output_scaler"].inverse_transform(scaled_output)


def _cross_validate(
    X: np.ndarray,
    Y: np.ndarray,
    variance_threshold: float,
    regressor_candidates: list[dict[str, Any]],
    random_state: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    n_splits = min(5, len(X))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates: list[dict[str, Any]] = []
    predictions_by_name: dict[str, np.ndarray] = {}

    for config in regressor_candidates:
        predictions = np.empty_like(Y)
        fold_errors: list[float] = []
        for train_idx, test_idx in splitter.split(X):
            pipeline = _fit_pipeline(X[train_idx], Y[train_idx], variance_threshold, config)
            fold_prediction = _predict_vector(pipeline, X[test_idx])
            predictions[test_idx] = fold_prediction
            scale = np.std(Y[train_idx], axis=0, ddof=0)
            scale[scale < 1e-12] = 1.0
            fold_errors.append(float(np.sqrt(np.mean(((fold_prediction - Y[test_idx]) / scale) ** 2))))
        score = float(np.mean(fold_errors))
        candidates.append({**config, "normalized_rmse": score})
        predictions_by_name[config["name"]] = predictions

    best_result = min(candidates, key=lambda item: item["normalized_rmse"])
    best_config = next(config for config in regressor_candidates if config["name"] == best_result["name"])
    return best_config, predictions_by_name[best_config["name"]], candidates


def _metrics(Y: np.ndarray, prediction: np.ndarray, n_scalars: int) -> dict[str, Any]:
    scalar_true = Y[:, :n_scalars]
    scalar_prediction = prediction[:, :n_scalars]
    cop_true = Y[:, n_scalars:]
    cop_prediction = prediction[:, n_scalars:]
    scalar_metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(SCALAR_NAMES):
        scalar_metrics[name] = {
            "mae": float(np.mean(np.abs(scalar_true[:, index] - scalar_prediction[:, index]))),
            "rmse": float(np.sqrt(np.mean((scalar_true[:, index] - scalar_prediction[:, index]) ** 2))),
            "r2": float(r2_score(scalar_true[:, index], scalar_prediction[:, index])),
        }
    return {
        "scalars": scalar_metrics,
        "cop": {
            "mae": float(np.mean(np.abs(cop_true - cop_prediction))),
            "rmse": float(np.sqrt(np.mean((cop_true - cop_prediction) ** 2))),
            "r2_variance_weighted": float(r2_score(cop_true, cop_prediction, multioutput="variance_weighted")),
        },
    }


def train_latent_surrogate(
    dataset: dict[str, Any],
    variance_threshold: float = 0.99,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    X = dataset["X"]
    Y = dataset["Y"]
    if len(X) < 6:
        raise ValueError("At least six unique reports are required for cross-validation")
    if not 0.5 <= variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be between 0.5 and 1.0")

    regressor_candidates = [
        {"name": "ridge_alpha_1", "kind": "ridge", "alpha": 1.0},
        {"name": "ridge_alpha_10", "kind": "ridge", "alpha": 10.0},
        {"name": "ridge_alpha_100", "kind": "ridge", "alpha": 100.0},
        {"name": "rbf_svr_C10_gamma_0.01", "kind": "rbf_svr", "C": 10.0, "gamma": 0.01, "epsilon": 0.05},
        {"name": "rbf_svr_C10_gamma_0.03", "kind": "rbf_svr", "C": 10.0, "gamma": 0.03, "epsilon": 0.05},
        {"name": "rbf_svr_C100_gamma_0.01", "kind": "rbf_svr", "C": 100.0, "gamma": 0.01, "epsilon": 0.05},
    ]
    best_config, oof_prediction, candidates = _cross_validate(
        X, Y, variance_threshold, regressor_candidates, random_state
    )
    fitted = _fit_pipeline(X, Y, variance_threshold, best_config)
    fitted_prediction = _predict_vector(fitted, X)

    current_index = SCALAR_NAMES.index("optimal_current_A")
    possible_boundary_samples = [
        {
            "file": dataset["filenames"][index],
            "flag": "possible_10A_current_search_boundary",
            "optimal_current_A": float(dataset["scalar_outputs"][index, current_index]),
        }
        for index in range(len(X))
        if dataset["scalar_outputs"][index, current_index] >= 9.999
    ]

    model = {
        "model_type": f"standardized_pca_{best_config['kind']}",
        "version": 2,
        "input_scaler": fitted["input_scaler"],
        "output_scaler": fitted["output_scaler"],
        "pca": fitted["pca"],
        "regressor": fitted["regressor"],
        "input_names": dataset["input_names"],
        "output_names": dataset["output_names"],
        "scalar_names": dataset["scalar_names"],
        "currents": dataset["currents"],
        "delta_t_values": dataset["delta_t_values"],
        "variance_threshold": variance_threshold,
        "latent_dim": fitted["latent_dim"],
        "regressor_config": best_config,
        "training_filenames": dataset["filenames"],
    }
    summary = {
        "source_file_count": dataset["source_file_count"],
        "unique_sample_count": len(X),
        "duplicate_files": dataset["duplicates"],
        "input_count": X.shape[1],
        "output_count": Y.shape[1],
        "scalar_output_count": len(SCALAR_NAMES),
        "cop_grid_shape": [len(dataset["delta_t_values"]), len(dataset["currents"])],
        "variance_threshold": variance_threshold,
        "latent_dim": fitted["latent_dim"],
        "retained_variance": float(np.sum(fitted["pca"].explained_variance_ratio_)),
        "explained_variance_ratio": fitted["all_explained_variance_ratio"].tolist(),
        "regressor_selection": candidates,
        "selected_regressor": best_config,
        "diagnostic_flags": possible_boundary_samples,
        "cross_validation": _metrics(Y, oof_prediction, len(SCALAR_NAMES)),
        "training_fit": _metrics(Y, fitted_prediction, len(SCALAR_NAMES)),
        "input_ranges": {
            name: {"min": float(X[:, index].min()), "max": float(X[:, index].max())}
            for index, name in enumerate(INPUT_NAMES)
        },
    }
    return model, summary, oof_prediction


def predict(model_or_path: dict[str, Any] | str | Path, X: np.ndarray) -> dict[str, np.ndarray]:
    """Predict scalar metrics, the COP surface, and latent coordinates."""
    model = joblib.load(model_or_path) if isinstance(model_or_path, (str, Path)) else model_or_path
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim == 1:
        X_array = X_array[None, :]
    if X_array.ndim != 2 or X_array.shape[1] != len(model["input_names"]):
        raise ValueError(f"X must have shape (n, {len(model['input_names'])})")

    latent = model["regressor"].predict(model["input_scaler"].transform(X_array))
    if latent.ndim == 1:
        latent = latent[:, None]
    output_scaled = model["pca"].inverse_transform(latent)
    output = model["output_scaler"].inverse_transform(output_scaled)
    n_scalars = len(model["scalar_names"])
    cop_shape = (len(X_array), len(model["delta_t_values"]), len(model["currents"]))
    return {
        "scalar_outputs": output[:, :n_scalars],
        "cop_surfaces": output[:, n_scalars:].reshape(cop_shape),
        "latent": latent,
        "output_vector": output,
    }


def save_dataset(dataset: dict[str, Any]) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DEFAULT_DATASET_PATH,
        X=dataset["X"],
        Y=dataset["Y"],
        scalar_outputs=dataset["scalar_outputs"],
        cop_surfaces=dataset["cop_surfaces"],
        filenames=np.asarray(dataset["filenames"]),
        input_names=np.asarray(dataset["input_names"]),
        scalar_names=np.asarray(dataset["scalar_names"]),
        output_names=np.asarray(dataset["output_names"]),
        currents=np.asarray(dataset["currents"]),
        delta_t_values=np.asarray(dataset["delta_t_values"]),
    )

    records = []
    for index, filename in enumerate(dataset["filenames"]):
        records.append(
            {
                "file": filename,
                "inputs": dict(zip(INPUT_NAMES, dataset["X"][index].tolist())),
                "scalars": dict(zip(SCALAR_NAMES, dataset["scalar_outputs"][index].tolist())),
                "cop_surface": dataset["cop_surfaces"][index].tolist(),
            }
        )
    document = {
        "input_names": INPUT_NAMES,
        "scalar_names": SCALAR_NAMES,
        "cop_axes": {
            "rows_delta_t_K": dataset["delta_t_values"],
            "columns_relative_current_I0": dataset["currents"],
        },
        "duplicates": dataset["duplicates"],
        "records": records,
    }
    DEFAULT_DATASET_JSON.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_diagnostics(
    dataset: dict[str, Any],
    model: dict[str, Any],
    summary: dict[str, Any],
    oof_prediction: np.ndarray,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    variance = np.asarray(summary["explained_variance_ratio"])
    cumulative = np.cumsum(variance)
    axes[0, 0].plot(np.arange(1, len(cumulative) + 1), cumulative, "o-", color="#176B87")
    axes[0, 0].axhline(summary["variance_threshold"], color="#C2413B", linestyle="--")
    axes[0, 0].axvline(summary["latent_dim"], color="#C2413B", linestyle=":")
    axes[0, 0].set(xlabel="Principal components", ylabel="Cumulative variance", ylim=(0, 1.02))
    axes[0, 0].set_title(f"PCA: {summary['latent_dim']} dimensions retain {summary['retained_variance']:.2%}")

    latent = model["pca"].transform(model["output_scaler"].transform(dataset["Y"]))
    second = latent[:, 1] if latent.shape[1] > 1 else np.zeros(len(latent))
    colors = dataset["X"][:, INPUT_NAMES.index("Tref_K")]
    scatter = axes[0, 1].scatter(latent[:, 0], second, c=colors, cmap="viridis", s=55, edgecolor="white")
    axes[0, 1].set(xlabel="Latent coordinate 1", ylabel="Latent coordinate 2")
    axes[0, 1].set_title("COMSOL report latent space")
    fig.colorbar(scatter, ax=axes[0, 1], label="Tref [K]")

    n_scalars = len(SCALAR_NAMES)
    cop_true = dataset["Y"][:, n_scalars:].reshape(-1)
    cop_pred = oof_prediction[:, n_scalars:].reshape(-1)
    low = min(cop_true.min(), cop_pred.min())
    high = max(cop_true.max(), cop_pred.max())
    axes[1, 0].scatter(cop_true, cop_pred, s=13, alpha=0.55, color="#176B87")
    axes[1, 0].plot([low, high], [low, high], "--", color="#C2413B")
    axes[1, 0].set(xlabel="True COP", ylabel="Cross-validated COP prediction")
    axes[1, 0].set_title(f"Out-of-fold COP, R2={summary['cross_validation']['cop']['r2_variance_weighted']:.3f}")

    cop_shape = dataset["cop_surfaces"].shape[1:]
    per_sample_mae = np.mean(
        np.abs(oof_prediction[:, n_scalars:].reshape((-1,) + cop_shape) - dataset["cop_surfaces"]),
        axis=(1, 2),
    )
    axes[1, 1].bar(np.arange(len(per_sample_mae)), per_sample_mae, color="#D58A35")
    axes[1, 1].set(xlabel="Unique report index", ylabel="COP MAE")
    axes[1, 1].set_title("Out-of-fold error by report")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DEFAULT_FIGURE_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--variance-threshold", type=float, default=0.99)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print(f"Parsing COMSOL reports from {args.report_dir}")
    dataset = extract_dataset(args.report_dir)
    print(
        f"Parsed {dataset['source_file_count']} files -> {len(dataset['filenames'])} unique samples "
        f"({len(dataset['duplicates'])} duplicate)"
    )
    save_dataset(dataset)

    model, summary, oof_prediction = train_latent_surrogate(
        dataset, variance_threshold=args.variance_threshold, random_state=args.random_state
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, DEFAULT_MODEL_PATH)
    DEFAULT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_diagnostics(dataset, model, summary, oof_prediction)

    print(f"Latent dimension: {summary['latent_dim']} ({summary['retained_variance']:.2%} retained variance)")
    print(f"Selected regressor: {summary['selected_regressor']['name']}")
    print(
        "Cross-validated COP: "
        f"MAE={summary['cross_validation']['cop']['mae']:.4f}, "
        f"R2={summary['cross_validation']['cop']['r2_variance_weighted']:.4f}"
    )
    print(f"Saved model: {DEFAULT_MODEL_PATH}")
    print(f"Saved dataset: {DEFAULT_DATASET_PATH}")
    print(f"Saved summary: {DEFAULT_SUMMARY_PATH}")
    print(f"Saved figure: {DEFAULT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
