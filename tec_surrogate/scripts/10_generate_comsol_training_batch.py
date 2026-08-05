"""Generate the next space-filling COMSOL report-training batch."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats.qmc import Sobol


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT / "outputs" / "comsol_training_batch_50.csv"
DEFAULT_SUMMARY = PROJECT / "outputs" / "comsol_training_batch_50_summary.json"
DEFAULT_GUIDE = PROJECT / "outputs" / "COMSOL_TRAINING_BATCH_50.md"

PARAMETER_NAMES = [
    "length_mm",
    "width_mm",
    "height_mm",
    "d_conductor_um",
    "d_ceramics_mm",
    "leg_length_mm",
    "leg_width_mm",
    "pitch_mm",
    "Tref_K",
]

# Intersection of the current COMSOL App input limits and the intended model domain.
LOWER = np.array([2.0, 1.5, 1.25, 50.0, 0.10, 0.50, 0.60, 0.25, 300.0])
UPPER = np.array([7.0, 5.0, 3.75, 150.0, 0.45, 1.20, 1.20, 0.75, 350.0])
ROUND_DIGITS = np.array([4, 4, 4, 2, 4, 4, 4, 4, 2])
CORE_BATCH_SIZE = 40
MIN_LEG_HEIGHT_MM = 0.20
MAX_N_PAIRS = 20
REQUIRED_PAIR_COVERAGE = {5: 2, 8: 2, 12: 2, 16: 2, 20: 2}


def _load_existing_inputs() -> np.ndarray:
    processed_path = PROJECT / "data" / "processed" / "comsol_report_dataset.npz"
    if processed_path.exists():
        with np.load(processed_path, allow_pickle=False) as dataset:
            inputs = np.asarray(dataset["X"], dtype=float)
        source_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]
        return inputs[:, source_indices]

    script_path = PROJECT / "scripts" / "09_train_from_reports.py"
    spec = importlib.util.spec_from_file_location("report_training", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {script_path}")
    spec.loader.exec_module(module)
    dataset = module.extract_dataset(PROJECT / "data" / "raw" / "doc")
    source_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    return np.asarray(dataset["X"][:, source_indices], dtype=float)


def geometry_metadata(values: np.ndarray) -> dict[str, float] | None:
    length, width, height, conductor_um, ceramics, leg_length, leg_width, pitch, _ = values
    leg_height = height - 2.0 * (conductor_um / 1000.0 + ceramics)
    if leg_height < MIN_LEG_HEIGHT_MM:
        return None

    raw_n_length = int(np.floor((length - 2.0 * pitch - leg_length) / (leg_length + pitch)) + 1)
    n_length = raw_n_length - raw_n_length % 2
    n_width = int(np.floor((width - 2.0 * pitch - leg_width) / (leg_width + pitch)) + 1)
    if n_length < 2 or n_width < 1:
        return None

    network_length = (leg_length + pitch) * n_length - pitch
    network_width = (leg_width + pitch) * n_width - pitch
    if network_length + 2.0 * pitch > length + 1e-9:
        return None
    if network_width + 2.0 * pitch > width + 1e-9:
        return None

    n_pairs = (n_length * n_width) // 2
    if not 1 <= n_pairs <= MAX_N_PAIRS:
        return None
    return {
        "expected_n_length": float(n_length),
        "expected_n_width": float(n_width),
        "expected_n_pairs": float(n_pairs),
        "expected_leg_height_mm": float(leg_height),
    }


def _round_values(values: np.ndarray) -> np.ndarray:
    return np.array(
        [round(float(value), int(digits)) for value, digits in zip(values, ROUND_DIGITS)],
        dtype=float,
    )


def generate_batch(
    n_samples: int = 50,
    seed: int = 20260716,
    candidate_power: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= n_samples <= 200:
        raise ValueError("n_samples must be between 1 and 200")

    sampler = Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=seed)
    normalized_pool = sampler.random_base2(m=candidate_power)
    physical_pool = LOWER + normalized_pool * (UPPER - LOWER)
    boundary_anchors = np.asarray(
        [
            [7.0, 5.0, 2.5, 100.0, 0.30, 0.50, 0.60, 0.25, 323.15],
            [6.9, 4.9, 3.2, 75.0, 0.20, 0.50, 0.60, 0.25, 340.0],
        ],
        dtype=float,
    )
    physical_pool = np.vstack((physical_pool, boundary_anchors))
    normalized_pool = np.vstack(
        (normalized_pool, (boundary_anchors - LOWER) / (UPPER - LOWER))
    )

    candidate_values: list[np.ndarray] = []
    candidate_metadata: list[dict[str, float]] = []
    candidate_normalized: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for normalized, physical in zip(normalized_pool, physical_pool):
        rounded = _round_values(physical)
        key = tuple(rounded)
        if key in seen:
            continue
        metadata = geometry_metadata(rounded)
        if metadata is None:
            continue
        seen.add(key)
        candidate_values.append(rounded)
        candidate_metadata.append(metadata)
        candidate_normalized.append((rounded - LOWER) / (UPPER - LOWER))

    if len(candidate_values) < n_samples:
        raise RuntimeError(f"Only {len(candidate_values)} feasible candidates for {n_samples} samples")

    pool_values = np.asarray(candidate_values)
    pool_normalized = np.asarray(candidate_normalized)
    pool_n_pairs = np.asarray(
        [int(metadata["expected_n_pairs"]) for metadata in candidate_metadata], dtype=int
    )
    existing = _load_existing_inputs()
    existing_normalized = np.clip((existing - LOWER) / (UPPER - LOWER), 0.0, 1.0)

    min_distance_sq = np.min(
        np.sum((pool_normalized[:, None, :] - existing_normalized[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    selected_indices: list[int] = []
    available = np.ones(len(pool_values), dtype=bool)

    def select_index(eligible: np.ndarray) -> None:
        scores = np.where(available & eligible, min_distance_sq, -np.inf)
        selected = int(np.argmax(scores))
        if not np.isfinite(scores[selected]):
            raise RuntimeError("Cannot satisfy required PN-pair coverage")
        selected_indices.append(selected)
        available[selected] = False
        distance_to_selected_sq = np.sum(
            (pool_normalized - pool_normalized[selected]) ** 2, axis=1
        )
        np.minimum(min_distance_sq, distance_to_selected_sq, out=min_distance_sq)

    for n_pairs, required_count in REQUIRED_PAIR_COVERAGE.items():
        for _ in range(min(required_count, n_samples - len(selected_indices))):
            select_index(pool_n_pairs == n_pairs)

    for _ in range(n_samples - len(selected_indices)):
        scores = np.where(available, min_distance_sq, -np.inf)
        select_index(np.isfinite(scores))

    rows: list[dict[str, Any]] = []
    for priority, selected in enumerate(selected_indices, start=1):
        values = pool_values[selected]
        metadata = candidate_metadata[selected]
        row: dict[str, Any] = {
            "sample_id": 99 + priority,
            "batch_group": "core_40" if priority <= min(CORE_BATCH_SIZE, n_samples) else "extension_10",
            "priority": priority,
        }
        row.update({name: float(value) for name, value in zip(PARAMETER_NAMES, values)})
        row["dT0_K"] = 50.0
        row.update(
            {
                "expected_n_pairs": int(metadata["expected_n_pairs"]),
                "expected_n_length": int(metadata["expected_n_length"]),
                "expected_n_width": int(metadata["expected_n_width"]),
                "expected_leg_height_mm": round(metadata["expected_leg_height_mm"], 5),
            }
        )
        rows.append(row)

    selected_normalized = pool_normalized[selected_indices]
    selected_to_existing = np.sqrt(
        np.min(
            np.sum(
                (selected_normalized[:, None, :] - existing_normalized[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
    )
    summary = {
        "method": "greedy_maximin_from_scrambled_sobol_pool",
        "seed": seed,
        "candidate_pool_size": int(len(normalized_pool)),
        "feasible_candidate_count": int(len(pool_values)),
        "selected_sample_count": n_samples,
        "core_sample_ids": [100, 99 + min(CORE_BATCH_SIZE, n_samples)],
        "extension_sample_ids": [140, 99 + n_samples] if n_samples > CORE_BATCH_SIZE else None,
        "parameter_names": PARAMETER_NAMES,
        "parameter_bounds": {
            name: {"min": float(low), "max": float(high)}
            for name, low, high in zip(PARAMETER_NAMES, LOWER, UPPER)
        },
        "selected_parameter_ranges": {
            name: {
                "min": float(pool_values[selected_indices, index].min()),
                "max": float(pool_values[selected_indices, index].max()),
            }
            for index, name in enumerate(PARAMETER_NAMES)
        },
        "constraints": {
            "minimum_leg_height_mm": MIN_LEG_HEIGHT_MM,
            "n_pairs": [1, MAX_N_PAIRS],
            "dT0_K": "50 (the current UI batch runner ignores this column)",
            "required_n_pair_coverage": REQUIRED_PAIR_COVERAGE,
        },
        "expected_n_pairs_distribution": dict(
            sorted(Counter(int(row["expected_n_pairs"]) for row in rows).items())
        ),
        "normalized_distance": {
            "minimum_selected_to_existing": float(selected_to_existing.min()),
            "median_selected_to_existing": float(np.median(selected_to_existing)),
            "minimum_between_selected": float(pdist(selected_normalized).min()) if n_samples > 1 else None,
        },
    }
    return rows, summary


def save_batch(rows: list[dict[str, Any]], summary: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    DEFAULT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    guide = f"""# COMSOL 新增训练批次

- CSV：`{csv_path.name}`
- 样本编号：`100-{99 + len(rows)}`
- 优先批次：`100-{99 + min(CORE_BATCH_SIZE, len(rows))}`
- 追加批次：`140-{99 + len(rows)}`（计算预算允许时）
- 固定工况：报告中的 COP 网格保持现有设置；`dT0_K` 列仅用于记录，当前批跑工具不会填写它。

在批跑工具中选择该 CSV。先运行 `100-139`，确认报告归档正常后，再运行 `140-149`。

`expected_n_pairs`、`expected_n_length`、`expected_n_width` 和
`expected_leg_height_mm` 是预检列，批跑工具会忽略，不需要手工删除。
"""
    DEFAULT_GUIDE.write_text(guide, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    rows, summary = generate_batch(args.count, args.seed)
    save_batch(rows, summary, args.output)
    print(f"Saved {len(rows)} space-filling samples: {args.output}")
    print(f"Expected PN-pair distribution: {summary['expected_n_pairs_distribution']}")
    print(
        "Minimum normalized distance to existing reports: "
        f"{summary['normalized_distance']['minimum_selected_to_existing']:.4f}"
    )


if __name__ == "__main__":
    main()
