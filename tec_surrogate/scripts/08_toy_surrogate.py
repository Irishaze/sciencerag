"""08_toy_surrogate.py — Toy latent space surrogate from 8 samples.

Proof of concept: PCA on sweep curves → Ridge regression from params → reconstruction.
With only 8 samples this is a feasibility demo, not a validated model.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, OUTPUTS, FIGURES, PARAM_NAMES

QUANTITIES = ["delta_T", "cooling_power", "input_power", "voltage"]
UNITS = {"delta_T": "K", "cooling_power": "W", "input_power": "W", "voltage": "V"}


def load_data():
    """Load all available samples."""
    files = sorted(RAW.glob("sample_*.npz"))
    params_list, curves_dict = [], {q: [] for q in QUANTITIES}
    ids = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        sid = int(data["sample_id"])
        # Check completeness
        if all(f"sweep_{q}" in data for q in QUANTITIES):
            params_list.append(data["params"])
            for q in QUANTITIES:
                arr = data[f"sweep_{q}"]
                curves_dict[q].append(np.nan_to_num(arr, nan=0.0))
            ids.append(sid)

    X = np.array(params_list)  # (N, 10)
    Y_curves = {q: np.array(curves_dict[q]) for q in QUANTITIES}  # each (N, 20)
    return X, Y_curves, ids


class ToyLatentSurrogate:
    """Simple PCA + Ridge surrogate for sweep curves."""

    def __init__(self, n_pc_per_quantity=2):
        self.n_pc = n_pc_per_quantity
        self.pca_basis = {}
        self.pca_mu = {}
        self.scaler_X = StandardScaler()
        self.regressors = {}

    def fit(self, X, Y_curves):
        """Fit PCA per quantity + Ridge regression."""
        X_scaled = self.scaler_X.fit_transform(X)

        all_scores = []
        for qty in QUANTITIES:
            Y = Y_curves[qty]  # (N, 20)
            mu = Y.mean(axis=0)
            Yc = Y - mu
            U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

            k = min(self.n_pc, len(S))
            self.pca_basis[qty] = Vt[:k]  # (k, 20)
            self.pca_mu[qty] = mu

            scores = Yc @ Vt[:k].T  # (N, k)
            all_scores.append(scores)

            print(f"  {qty}: μ_range=[{mu.min():.3g}, {mu.max():.3g}], "
                  f"σ={S[:k].round(3)}")

        # Combined latent = concat of per-quantity PCA scores
        latent_all = np.concatenate(all_scores, axis=1)  # (N, total_k)
        print(f"  Total latent dim: {latent_all.shape[1]}")

        # Train ridge regressor per latent dimension
        for j in range(latent_all.shape[1]):
            ridge = Ridge(alpha=0.1)
            ridge.fit(X_scaled, latent_all[:, j])
            self.regressors[j] = ridge

        self.n_latent = latent_all.shape[1]
        self.k_per_qty = {q: min(self.n_pc, Y_curves[q].shape[1]) for q in QUANTITIES}

    def predict(self, X_new):
        """Predict sweep curves for new parameters."""
        X_scaled = self.scaler_X.transform(X_new)
        n = len(X_new)

        # Predict latent
        latent_pred = np.column_stack([
            self.regressors[j].predict(X_scaled) for j in range(self.n_latent)
        ])

        # Reconstruct per quantity
        curves_pred = {}
        offset = 0
        for qty in QUANTITIES:
            k = self.k_per_qty[qty]
            scores_q = latent_pred[:, offset:offset + k]
            recon = scores_q @ self.pca_basis[qty] + self.pca_mu[qty]
            curves_pred[qty] = recon
            offset += k
        return curves_pred

    def score(self, X, Y_curves):
        """R2 per quantity."""
        pred = self.predict(X)
        scores = {}
        for qty in QUANTITIES:
            ss_res = np.sum((Y_curves[qty] - pred[qty]) ** 2)
            ss_tot = np.sum((Y_curves[qty] - Y_curves[qty].mean()) ** 2)
            scores[qty] = 1 - ss_res / (ss_tot + 1e-12)
        return scores


def plot_predictions(model, X, Y_true, sample_ids, output_dir):
    """Plot true vs predicted curves for all samples."""
    Y_pred = model.predict(X)
    i0 = np.logspace(np.log10(0.1), np.log10(5.0), 20)

    n_show = min(4, len(X))
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for row, sample_idx in enumerate(range(n_show)):
        sid = sample_ids[sample_idx]
        for col, qty in enumerate(QUANTITIES):
            ax = axes[row, col]
            ax.plot(i0, Y_true[qty][sample_idx], "o-", color="steelblue",
                     markersize=3, label="True")
            ax.plot(i0, Y_pred[qty][sample_idx], "s--", color="darkorange",
                     markersize=3, label="Pred")
            ax.set_xscale("log")
            ax.set_xlabel("I0 [A]")
            ax.set_ylabel(f"{qty} [{UNITS[qty]}]")
            ax.set_title(f"Sample {sid}: {qty}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 3:
                ax.legend(fontsize=7)

    fig.suptitle("Toy Surrogate: True vs Predicted Sweep Curves (8-sample fit)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "toy_surrogate_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_space(model, X, sample_ids, output_dir):
    """Visualize the 2D latent space."""
    X_scaled = model.scaler_X.transform(X)

    latent_all = np.column_stack([
        model.regressors[j].predict(X_scaled) for j in range(model.n_latent)
    ])

    # PCA the latent for 2D visualization
    from sklearn.decomposition import PCA
    latent_2d = PCA(n_components=2).fit_transform(latent_all)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=range(len(X)),
                          cmap="viridis", s=80, edgecolors="k", linewidth=0.5)
    for i, sid in enumerate(sample_ids):
        ax.annotate(str(sid), (latent_2d[i, 0], latent_2d[i, 1]),
                     fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.set_title("Toy Latent Space (8 TEC Designs)")
    plt.colorbar(scatter, label="Sample index")
    fig.tight_layout()
    fig.savefig(output_dir / "toy_latent_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading data...")
    X, Y_curves, ids = load_data()
    print(f"Samples: {len(ids)}, Params: {X.shape[1]}, Curve dims: {[Y_curves[q].shape for q in QUANTITIES]}")

    # Fit model (1 PC per quantity — based on POD analysis)
    print("\nFitting toy surrogate (1 PC per quantity)...")
    model = ToyLatentSurrogate(n_pc_per_quantity=1)
    model.fit(X, Y_curves)

    # Score (training R2 — will be near-perfect with 8 samples and 4 params)
    print("\nTraining R2:")
    r2 = model.score(X, Y_curves)
    for q, v in r2.items():
        print(f"  {q}: {v:.4f}")

    # Predictions
    plot_predictions(model, X, Y_curves, ids, FIGURES)
    plot_latent_space(model, X, ids, FIGURES)

    # Save model
    import joblib
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUTS / "toy_surrogate.joblib")
    print(f"\nSaved: {OUTPUTS / 'toy_surrogate.joblib'}")
    print(f"Latent dim: {model.n_latent} (4 quantities × 1 PC each)")
    print("\nReady for more data — re-fit when batch completes.")


if __name__ == "__main__":
    main()
