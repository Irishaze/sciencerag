"""Load exported COMSOL component cases as PyTorch graph samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .graph import ComponentGraph, InterfaceSamples


def _face_axis_points(lower: float, upper: float, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("samples_per_axis must be positive")
    if count == 1:
        return np.asarray([(lower + upper) / 2.0])
    step = (upper - lower) / count
    return lower + (np.arange(count) + 0.5) * step


def load_component_interfaces(
    path: str | Path,
    samples_per_axis: int = 3,
    thermal_resistance: float = 0.0,
    electrical_resistance: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> InterfaceSamples:
    """Convert exported contacts to matched local points on both face sides.

    Every contact is sampled on a cell-centred ``samples_per_axis`` square
    grid over its physical overlap. Electrical constraints are enabled only
    when both components carry an electric-potential field.
    """

    with np.load(path, allow_pickle=False) as data:
        contacts = np.asarray(data["contacts"], dtype=float).reshape(-1, 7)
        metadata = json.loads(str(data["metadata_json"]))
        boxes = np.asarray(
            [component["bbox"] for component in metadata["components"]], dtype=float
        ).reshape(-1, 3, 2)
        field_mask = np.asarray(data["node_field_mask"], dtype=bool)

    source_components: list[int] = []
    target_components: list[int] = []
    source_coords: list[np.ndarray] = []
    target_coords: list[np.ndarray] = []
    source_normals: list[np.ndarray] = []
    electrical_masks: list[bool] = []

    for row in contacts:
        source, target, axis = (int(row[0]), int(row[1]), int(row[2]))
        normal = np.asarray(row[3:6], dtype=float)
        other_axes = [item for item in range(3) if item != axis]
        overlap = [
            (
                max(boxes[source, item, 0], boxes[target, item, 0]),
                min(boxes[source, item, 1], boxes[target, item, 1]),
            )
            for item in other_axes
        ]
        if any(upper <= lower for lower, upper in overlap):
            raise ValueError(f"contact {source}->{target} has no face overlap")

        first = _face_axis_points(*overlap[0], samples_per_axis)
        second = _face_axis_points(*overlap[1], samples_per_axis)
        for first_value in first:
            for second_value in second:
                physical = np.empty(3, dtype=float)
                physical[other_axes] = (first_value, second_value)
                physical[axis] = (
                    boxes[source, axis, 1]
                    if normal[axis] > 0
                    else boxes[source, axis, 0]
                )
                source_local = (
                    (physical - boxes[source, :, 0])
                    / (boxes[source, :, 1] - boxes[source, :, 0])
                )
                physical[axis] = (
                    boxes[target, axis, 0]
                    if normal[axis] > 0
                    else boxes[target, axis, 1]
                )
                target_local = (
                    (physical - boxes[target, :, 0])
                    / (boxes[target, :, 1] - boxes[target, :, 0])
                )
                source_components.append(source)
                target_components.append(target)
                source_coords.append(source_local)
                target_coords.append(target_local)
                source_normals.append(normal)
                electrical_masks.append(bool(field_mask[source, 1] and field_mask[target, 1]))

    count = len(source_components)
    return InterfaceSamples(
        source_components=torch.as_tensor(source_components, dtype=torch.long),
        target_components=torch.as_tensor(target_components, dtype=torch.long),
        source_coords=torch.as_tensor(np.asarray(source_coords).reshape(-1, 3), dtype=dtype),
        target_coords=torch.as_tensor(np.asarray(target_coords).reshape(-1, 3), dtype=dtype),
        source_normals=torch.as_tensor(np.asarray(source_normals).reshape(-1, 3), dtype=dtype),
        thermal_resistance=torch.full((count,), thermal_resistance, dtype=dtype),
        electrical_resistance=torch.full((count,), electrical_resistance, dtype=dtype),
        thermal_mask=torch.ones(count, dtype=torch.bool),
        electrical_mask=torch.as_tensor(electrical_masks, dtype=torch.bool),
    ).validate()


def load_component_case(
    path: str | Path,
    solution_index: int,
    dtype: torch.dtype = torch.float32,
) -> ComponentGraph:
    """Load one solution from a component-case ``.npz`` export."""

    with np.load(path, allow_pickle=False) as data:
        temperature = np.asarray(data["temperature"], dtype=float)
        potential = np.asarray(data["potential"], dtype=float)
        if not 0 <= solution_index < len(temperature):
            raise IndexError(
                f"solution_index {solution_index} is outside [0, {len(temperature)})"
            )
        targets = np.column_stack(
            (temperature[solution_index], potential[solution_index])
        )
        def solution_property(name: str) -> np.ndarray:
            values = np.asarray(data[name])
            return values[solution_index] if values.ndim == 2 else values

        graph = ComponentGraph(
            node_features=torch.as_tensor(data["node_features"], dtype=dtype),
            node_types=torch.as_tensor(data["node_types"], dtype=torch.long),
            node_field_mask=torch.as_tensor(data["node_field_mask"], dtype=torch.bool),
            node_sizes=torch.as_tensor(data["node_sizes"], dtype=dtype),
            thermal_conductivity=torch.as_tensor(
                solution_property("thermal_conductivity"), dtype=dtype
            ),
            electrical_conductivity=torch.as_tensor(
                solution_property("electrical_conductivity"), dtype=dtype
            ),
            seebeck_coefficient=torch.as_tensor(
                solution_property("seebeck_coefficient"), dtype=dtype
            ),
            edge_index=torch.as_tensor(data["edge_index"], dtype=torch.long),
            edge_features=torch.as_tensor(data["edge_features"], dtype=dtype),
            global_features=torch.as_tensor(
                data["global_features"][solution_index : solution_index + 1],
                dtype=dtype,
            ),
            node_graph=torch.zeros(len(data["node_features"]), dtype=torch.long),
            probe_coords=torch.as_tensor(data["probe_coords"], dtype=dtype),
            probe_components=torch.as_tensor(
                data["probe_components"], dtype=torch.long
            ),
            probe_targets=torch.as_tensor(targets, dtype=dtype),
            probe_target_mask=torch.as_tensor(
                data["probe_target_mask"], dtype=torch.bool
            ),
        )
    return graph.validate()
