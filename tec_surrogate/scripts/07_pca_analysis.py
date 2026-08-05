"""07_pca_analysis.py — PCA/POD analysis on sweep curves to estimate latent dimension.

Phase 2/3: Determine effective latent space dimensionality for each physical quantity.
Uses only training samples (first 8 from train[:8]).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, DATA, OUTPUTS, FIGURES

# Physical quantities in sweep curves
QUANTITIES = ["delta_T", "cooling_power", "input_power", "voltage"]
UNITS = {"delta_T": "K", "cooling_power": "W", "input_power": "W", "voltage": "V"}


def load_samples(sample_ids):
    """Load sweep curves from sample .npz files."""
    curves = {q: [] for q in QUANTITIES}
    params = []
    valid_ids = []

    for sid in sample_ids:
        path = RAW / f"sample_{sid:04d}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        valid = True
        for q in QUANTITIES:
            key = f"sweep_{q}"
            if key not in data:
                valid = False
                break
            arr = data[key]
            if np.all(~np.isfinite(arr)) or len(arr) == 0:
                valid = False
                break
        if not valid:
            continue

        for q in QUANTITIES:
            curves[q].append(data[f"sweep_{q}"])
        params.append(data["params"])
        valid_ids.append(sid)

    return (np.array(params), {q: np.array(curves[q]) for q in QUANTITIES}, valid_ids)


def pca_analysis(X, quantity_name, unit, train_idx=None):
    """Run PCA and return results.

    X: (N, D) — N samples, D = n_current_points
    """
    if train_idx is None:
        train_idx = np.arange(len(X))

    X_train = X[train_idx]

    # Center
    mu = X_train.mean(axis=0)

    # SVD
    Xc = X_train - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Cumulative variance
    total_var = np.sum(S ** 2)
    cum_var = np.cumsum(S ** 2) / total_var

    # Find PCs needed for 95%, 99%, 99.9%
    k_95 = int(np.searchsorted(cum_var, 0.95) + 1)
    k_99 = int(np.searchsorted(cum_var, 0.99) + 1)
    k_999 = int(np.searchsorted(cum_var, 0.999) + 1)

    # Reconstruct with k PCs and compute errors
    errors = {}
    for k_label, k in [("k=1", 1), ("k=2", 2), ("k=3", 3),
                        ("k_95", k_95), ("k_99", k_99)]:
        scores = Xc @ Vt[:k].T
        X_recon = scores @ Vt[:k] + mu
        mae = np.mean(np.abs(X - X_recon))
        max_err = np.max(np.abs(X - X_recon))
        errors[k_label] = {"mae": float(mae), "max_err": float(max_err),
                            "mae_unit": f"{mae:.4g} {unit}", "k": k}

    return {
        "quantity": quantity_name,
        "unit": unit,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "singular_values": S.tolist()[:10],
        "cumulative_variance": cum_var.tolist(),
        "k_95": int(k_95),
        "k_99": int(k_99),
        "k_999": int(k_999),
        "reconstruction_errors": errors,
        "modes": Vt[:min(5, len(Vt))].tolist(),  # First 5 modes
    }


def plot_pca(results, output_dir):
    """Plot PCA variance curves for all quantities."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for ax, (qty, res) in zip(axes.flat, results.items()):
        cum_var = res["cumulative_variance"]
        n_pcs = min(20, len(cum_var))
        ax.plot(range(1, n_pcs + 1), cum_var[:n_pcs], "o-", color="steelblue", markersize=4)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
        ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="99%")

        # Annotate
        ax.annotate(f"k95={res['k_95']}", xy=(res["k_95"], 0.95),
                     xytext=(res["k_95"] + 1, 0.92),
                     arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
        ax.annotate(f"k99={res['k_99']}", xy=(res["k_99"], 0.99),
                     xytext=(res["k_99"] + 1, 0.96),
                     arrowprops=dict(arrowstyle="->", color="darkred"), color="darkred", fontsize=9)

        ax.set_xlabel("Number of PCs")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title(f"{qty} [{res['unit']}]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("POD Analysis: Sweep Curve Latent Dimensions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "pca_variance_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'pca_variance_curves.png'}")


def plot_modes(results, output_dir):
    """Plot first 3 POD modes for each quantity."""
    for qty, res in results.items():
        fig, axes = plt.subplots(1, min(3, len(res["modes"])), figsize=(12, 3.5))
        if not hasattr(axes, "__len__"):
            axes = [axes]

        for i, ax in enumerate(axes):
            mode = np.array(res["modes"][i])
            # Mode shape across I0 sweep points
            i0_points = np.logspace(np.log10(0.1), np.log10(5.0), len(mode))
            ax.plot(i0_points, mode, "o-", markersize=3, color="steelblue")
            ax.set_xlabel("I0 [A]")
            ax.set_xscale("log")
            ax.set_title(f"{qty} — Mode {i + 1}")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"POD Modes: {qty} [{res['unit']}]", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"modes_{qty}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved mode plots to: {output_dir}")


def main():
    # Load all available samples
    raw_files = sorted(RAW.glob("sample_*.npz"))
    all_ids = [int(f.stem.split("_")[1]) for f in raw_files]

    if len(all_ids) < 3:
        print(f"Need at least 3 samples, have {len(all_ids)}. Run more batch samples first.")
        return

    print(f"Loading {len(all_ids)} samples: {all_ids}")
    params, curves, valid_ids = load_samples(all_ids)
    print(f"Valid samples with complete sweep data: {len(valid_ids)}")
    print(f"Parameter shape: {params.shape}")
    for q in QUANTITIES:
        print(f"  {q}: {curves[q].shape}")

    # Run PCA per quantity
    results = {}
    for qty in QUANTITIES:
        X = curves[qty]  # (N, 20)
        # Remove clearly anomalous first-point values (I0=0.1A sometimes has solver artifacts)
        # Use all points for now
        res = pca_analysis(X, qty, UNITS[qty])
        results[qty] = res

        print(f"\n{qty} [{UNITS[qty]}]:")
        print(f"  k_95 = {res['k_95']}, k_99 = {res['k_99']}, k_999 = {res['k_999']}")
        for k_label, err in res["reconstruction_errors"].items():
            if k_label in ("k=1", "k_95", "k_99"):
                print(f"  {k_label} (k={err['k']}): MAE={err['mae_unit']}, max_err={err['max_err']:.4g}")

    # Plot
    plot_pca(results, FIGURES)
    plot_modes(results, FIGURES)

    # Save numerical results
    import json
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS / "pca_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved numerical results to: {OUTPUTS / 'pca_results.json'}")

    # Summary
    print("\n=== Latent Dimension Summary ===")
    print(f"Based on {len(valid_ids)} training samples:")
    for qty in QUANTITIES:
        k99 = results[qty]["k_99"]
        print(f"  {qty:20s}: k_99 = {k99:2d} PCs for 99% variance")
    total_k = sum(results[q]["k_99"] for q in QUANTITIES)
    print(f"  {'Total (combined)':20s}: {total_k} PCs")
    print(f"\nNote: 8 samples → max {len(valid_ids)-1} meaningful PCs. "
          f"Need 160 samples for reliable estimate.")


if __name__ == "__main__":
    main()
