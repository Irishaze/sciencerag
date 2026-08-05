"""Inference service for the hierarchical 1-20 pair component-field model."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .model import ComponentGraphSepONet
from .multi_pair import MAX_SUPPORTED_PAIRS, MultiPairLayout, compose_multi_pair_graph
from .real_data import load_component_case


class MultiPairFieldService:
    """Load the composed-field checkpoint and return UI-ready TEC summaries."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        case_path: str | Path,
        summary_path: str | Path,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.case_path = Path(case_path)
        self.summary_path = Path(summary_path)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model = ComponentGraphSepONet(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.field_offset = checkpoint["field_offset"]
        self.field_scale = checkpoint["field_scale"]
        self.holdout_pair_counts = set(checkpoint["holdout_pair_counts"])
        self.summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.base_graphs = [load_component_case(self.case_path, index) for index in range(11)]
        self.currents = np.asarray(
            [float(graph.global_features[0, 0]) for graph in self.base_graphs]
        )
        self.reference_hot_temperature = float(
            self.base_graphs[0].global_features[0, 1]
        )
        self.base_temperature_min = np.asarray(
            [float(graph.probe_targets[:, 0].min()) for graph in self.base_graphs]
        )
        self.base_temperature_max = np.asarray(
            [float(graph.probe_targets[:, 0].max()) for graph in self.base_graphs]
        )
        self.base_potential_span = np.asarray(
            [
                float(
                    graph.probe_targets[:, 1][graph.probe_target_mask[:, 1]].max()
                    - graph.probe_targets[:, 1][graph.probe_target_mask[:, 1]].min()
                )
                for graph in self.base_graphs
            ]
        )

    def metadata(self) -> dict[str, Any]:
        validation = self.summary["validation"]["unseen_pair_and_current"]
        return {
            "model_type": "hierarchical_component_graph_seponet",
            "pair_count": {"min": 1, "max": MAX_SUPPORTED_PAIRS, "default": 6},
            "current_A": {
                "min": round(float(self.currents.min()), 1),
                "max": round(float(self.currents.max()), 1),
                "step": 0.1,
                "default": 0.5,
            },
            "hot_temperature_K": {
                "min": 300.0,
                "max": 350.0,
                "step": 0.05,
                "default": round(self.reference_hot_temperature, 2),
            },
            "validation": {
                "temperature_rmse_K": validation["temperature_rmse_K"],
                "potential_rmse_V": validation["potential_rmse_V"],
                "holdout_pair_counts": sorted(self.holdout_pair_counts),
            },
            "calibration": {
                "direct_comsol_pair_counts": [1],
                "composed_pair_counts": [2, MAX_SUPPORTED_PAIRS],
            },
        }

    @staticmethod
    def _profile(
        prediction: torch.Tensor,
        graph: Any,
        layout: MultiPairLayout,
        pair_index: int,
        role: str,
    ) -> list[dict[str, float]]:
        pair_mask = layout.probe_pair == pair_index
        role_mask = torch.tensor(
            [value == role for value in layout.probe_role], dtype=torch.bool
        )
        indices = torch.nonzero(pair_mask & role_mask).flatten()
        z_values = torch.round(graph.probe_coords[indices, 2] * 1.0e6) / 1.0e6
        rows = []
        for z_value in torch.unique(z_values, sorted=True):
            group = indices[z_values == z_value]
            rows.append(
                {
                    "z": float(z_value),
                    "temperature_K": float(prediction[group, 0].mean()),
                    "potential_V": float(prediction[group, 1].mean()),
                }
            )
        return rows

    def predict(
        self, n_pairs: int, current_A: float, hot_temperature_K: float
    ) -> dict[str, Any]:
        if isinstance(n_pairs, bool) or not isinstance(n_pairs, (int, np.integer)):
            raise ValueError("n_pairs must be an integer")
        if not 1 <= int(n_pairs) <= MAX_SUPPORTED_PAIRS:
            raise ValueError(f"n_pairs must be between 1 and {MAX_SUPPORTED_PAIRS}")
        for name, value in (
            ("current_A", current_A),
            ("hot_temperature_K", hot_temperature_K),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.currents.min() <= current_A <= self.currents.max():
            raise ValueError(
                f"current_A must be between {self.currents.min():.1f} and "
                f"{self.currents.max():.1f}"
            )
        if not 300.0 <= hot_temperature_K <= 350.0:
            raise ValueError("hot_temperature_K must be between 300 and 350")

        started = time.perf_counter()
        nearest_solution = int(np.argmin(np.abs(self.currents - current_A)))
        graph, layout = compose_multi_pair_graph(
            self.base_graphs[nearest_solution],
            int(n_pairs),
            current=float(current_A),
            max_probes_per_component=48,
        )
        with torch.no_grad():
            prediction = self.model(graph) * self.field_scale + self.field_offset
        conductive = graph.probe_target_mask[:, 1]
        temperature_shift = hot_temperature_K - self.reference_hot_temperature
        desired_temperature_min = float(
            np.interp(current_A, self.currents, self.base_temperature_min)
            + temperature_shift
        )
        desired_temperature_max = float(
            np.interp(current_A, self.currents, self.base_temperature_max)
            + temperature_shift
        )
        predicted_temperature_min = prediction[:, 0].min()
        predicted_temperature_span = (
            prediction[:, 0].max() - predicted_temperature_min
        ).clamp_min(1.0e-8)
        prediction[:, 0] = desired_temperature_min + (
            (prediction[:, 0] - predicted_temperature_min)
            / predicted_temperature_span
            * (desired_temperature_max - desired_temperature_min)
        )

        desired_pair_voltage = float(
            np.interp(current_A, self.currents, self.base_potential_span)
        )
        for pair_index in range(n_pairs):
            pair_indices = torch.nonzero(
                (layout.probe_pair == pair_index) & conductive
            ).flatten()
            pair_values = prediction[pair_indices, 1]
            pair_minimum = pair_values.min()
            pair_span = (pair_values.max() - pair_minimum).clamp_min(1.0e-8)
            prediction[pair_indices, 1] = (
                (pair_values - pair_minimum)
                / pair_span
                * desired_pair_voltage
                + pair_index * desired_pair_voltage
            )

        temperature = prediction[:, 0]
        potential = prediction[:, 1][conductive]
        pairs = []
        profiles: dict[str, dict[str, list[dict[str, float]]]] = {}
        leg_probe_mask = torch.tensor(
            [role in {"p_leg", "n_leg"} for role in layout.probe_role],
            dtype=torch.bool,
        )
        for pair_index in range(n_pairs):
            indices = torch.nonzero(layout.probe_pair == pair_index).flatten()
            leg_indices = torch.nonzero(
                (layout.probe_pair == pair_index) & leg_probe_mask
            ).flatten()
            local_z = graph.probe_coords[leg_indices, 2]
            hot_indices = leg_indices[local_z <= 0.25]
            cold_indices = leg_indices[local_z >= 0.75]
            pair_conductive = indices[graph.probe_target_mask[indices, 1]]
            pair_potential = prediction[pair_conductive, 1]
            pairs.append(
                {
                    "pair": pair_index + 1,
                    "row": pair_index // layout.columns,
                    "column": pair_index % layout.columns,
                    "cold_temperature_K": float(prediction[cold_indices, 0].mean()),
                    "hot_temperature_K": float(prediction[hot_indices, 0].mean()),
                    "voltage_drop_V": float(pair_potential.max() - pair_potential.min()),
                    "potential_midpoint_V": float(pair_potential.mean()),
                }
            )
            profiles[str(pair_index + 1)] = {
                "p_leg": self._profile(
                    prediction, graph, layout, pair_index, "p_leg"
                ),
                "n_leg": self._profile(
                    prediction, graph, layout, pair_index, "n_leg"
                ),
            }

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        calibration = (
            "one_pair_comsol_anchor"
            if n_pairs == 1
            else "composed_topology_pending_multipair_comsol"
        )
        return {
            "inputs": {
                "n_pairs": int(n_pairs),
                "current_A": float(current_A),
                "hot_temperature_K": float(hot_temperature_K),
            },
            "layout": {"rows": layout.rows, "columns": layout.columns},
            "metrics": {
                "minimum_temperature_K": float(temperature.min()),
                "maximum_temperature_K": float(temperature.max()),
                "temperature_span_K": float(temperature.max() - temperature.min()),
                "terminal_voltage_V": float(potential.max() - potential.min()),
                "inference_ms": elapsed_ms,
            },
            "pairs": pairs,
            "profiles": profiles,
            "calibration": calibration,
            "pair_count_was_held_out": int(n_pairs) in self.holdout_pair_counts,
        }
