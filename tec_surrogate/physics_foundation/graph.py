"""Tensor containers for variable-topology thermoelectric component graphs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Sequence
from typing import Self

import torch


Tensor = torch.Tensor


@dataclass
class ComponentGraph:
    """One or more TEC component graphs packed into flat tensors.

    Coordinates are local to each component and must lie in ``[0, 1]^3``.
    ``node_sizes`` maps local derivatives to physical derivatives. All material
    properties and sizes must use one consistent unit system.
    """

    node_features: Tensor
    node_types: Tensor
    node_field_mask: Tensor
    node_sizes: Tensor
    thermal_conductivity: Tensor
    electrical_conductivity: Tensor
    seebeck_coefficient: Tensor
    edge_index: Tensor
    edge_features: Tensor
    global_features: Tensor
    node_graph: Tensor
    probe_coords: Tensor
    probe_components: Tensor
    probe_targets: Tensor | None = None
    probe_target_mask: Tensor | None = None

    def validate(self) -> Self:
        n_nodes = self.node_features.shape[0]
        if self.node_features.ndim != 2:
            raise ValueError("node_features must have shape (N, F)")
        if self.node_types.shape != (n_nodes,):
            raise ValueError("node_types must have shape (N,)")
        if self.node_field_mask.shape != (n_nodes, 2):
            raise ValueError("node_field_mask must have shape (N, 2)")
        if self.node_sizes.shape != (n_nodes, 3):
            raise ValueError("node_sizes must have shape (N, 3)")
        if torch.any(self.node_sizes <= 0):
            raise ValueError("node_sizes must be positive")
        for name in (
            "thermal_conductivity",
            "electrical_conductivity",
            "seebeck_coefficient",
            "node_graph",
        ):
            if getattr(self, name).shape != (n_nodes,):
                raise ValueError(f"{name} must have shape (N,)")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if self.edge_features.ndim != 2 or self.edge_features.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_features must have shape (E, Fe)")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= n_nodes
        ):
            raise ValueError("edge_index refers to a missing node")
        if self.global_features.ndim != 2:
            raise ValueError("global_features must have shape (B, G)")
        if n_nodes and int(self.node_graph.max()) >= len(self.global_features):
            raise ValueError("node_graph refers to a missing graph")
        if self.probe_coords.ndim != 2 or self.probe_coords.shape[1] != 3:
            raise ValueError("probe_coords must have shape (P, 3)")
        if self.probe_components.shape != (len(self.probe_coords),):
            raise ValueError("probe_components must have shape (P,)")
        if self.probe_components.numel() and (
            int(self.probe_components.min()) < 0 or int(self.probe_components.max()) >= n_nodes
        ):
            raise ValueError("probe_components refers to a missing node")
        if torch.any(self.probe_coords < 0) or torch.any(self.probe_coords > 1):
            raise ValueError("probe_coords must lie in [0, 1]")
        if self.probe_targets is not None and self.probe_targets.shape != (len(self.probe_coords), 2):
            raise ValueError("probe_targets must have shape (P, 2) for temperature and potential")
        if self.probe_target_mask is not None and self.probe_target_mask.shape != (len(self.probe_coords), 2):
            raise ValueError("probe_target_mask must have shape (P, 2)")
        if (self.probe_targets is None) != (self.probe_target_mask is None):
            raise ValueError("probe_targets and probe_target_mask must be provided together")
        return self

    def to(self, device: torch.device | str) -> Self:
        values = {
            field.name: (
                getattr(self, field.name).to(device)
                if isinstance(getattr(self, field.name), Tensor)
                else getattr(self, field.name)
            )
            for field in fields(self)
        }
        return type(self)(**values)


@dataclass
class InterfaceSamples:
    """Matched points on both sides of material interfaces."""

    source_components: Tensor
    target_components: Tensor
    source_coords: Tensor
    target_coords: Tensor
    source_normals: Tensor
    thermal_resistance: Tensor
    electrical_resistance: Tensor
    thermal_mask: Tensor | None = None
    electrical_mask: Tensor | None = None

    def validate(self) -> Self:
        count = len(self.source_components)
        if self.target_components.shape != (count,):
            raise ValueError("target_components must have shape (Q,)")
        for name in ("source_coords", "target_coords", "source_normals"):
            if getattr(self, name).shape != (count, 3):
                raise ValueError(f"{name} must have shape (Q, 3)")
        for name in ("thermal_resistance", "electrical_resistance"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape (Q,)")
        for name in ("thermal_mask", "electrical_mask"):
            value = getattr(self, name)
            if value is not None and value.shape != (count,):
                raise ValueError(f"{name} must have shape (Q,)")
        return self

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            **{
                field.name: (
                    getattr(self, field.name).to(device)
                    if isinstance(getattr(self, field.name), Tensor)
                    else getattr(self, field.name)
                )
                for field in fields(self)
            }
        )


@dataclass
class BoundarySamples:
    """Dirichlet observations or hard boundary targets for local components."""

    components: Tensor
    coords: Tensor
    targets: Tensor
    mask: Tensor

    def validate(self) -> Self:
        count = len(self.components)
        if self.coords.shape != (count, 3):
            raise ValueError("coords must have shape (Q, 3)")
        if self.targets.shape != (count, 2) or self.mask.shape != (count, 2):
            raise ValueError("targets and mask must have shape (Q, 2)")
        return self

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


def batch_component_graphs(graphs: Sequence[ComponentGraph]) -> ComponentGraph:
    """Pack variable-size component graphs for one vectorized model call."""

    if not graphs:
        raise ValueError("at least one graph is required")
    validated = [graph.validate() for graph in graphs]
    has_targets = [graph.probe_targets is not None for graph in validated]
    if any(has_targets) and not all(has_targets):
        raise ValueError("either every graph or no graph must provide probe_targets")

    edge_indices = []
    probe_components = []
    node_graph = []
    node_offset = 0
    graph_offset = 0
    for graph in validated:
        edge_indices.append(graph.edge_index + node_offset)
        probe_components.append(graph.probe_components + node_offset)
        node_graph.append(graph.node_graph + graph_offset)
        node_offset += len(graph.node_features)
        graph_offset += len(graph.global_features)

    return ComponentGraph(
        node_features=torch.cat([graph.node_features for graph in validated]),
        node_types=torch.cat([graph.node_types for graph in validated]),
        node_field_mask=torch.cat([graph.node_field_mask for graph in validated]),
        node_sizes=torch.cat([graph.node_sizes for graph in validated]),
        thermal_conductivity=torch.cat(
            [graph.thermal_conductivity for graph in validated]
        ),
        electrical_conductivity=torch.cat(
            [graph.electrical_conductivity for graph in validated]
        ),
        seebeck_coefficient=torch.cat(
            [graph.seebeck_coefficient for graph in validated]
        ),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat([graph.edge_features for graph in validated]),
        global_features=torch.cat([graph.global_features for graph in validated]),
        node_graph=torch.cat(node_graph),
        probe_coords=torch.cat([graph.probe_coords for graph in validated]),
        probe_components=torch.cat(probe_components),
        probe_targets=(
            torch.cat([graph.probe_targets for graph in validated if graph.probe_targets is not None])
            if all(has_targets)
            else None
        ),
        probe_target_mask=(
            torch.cat(
                [
                    graph.probe_target_mask
                    for graph in validated
                    if graph.probe_target_mask is not None
                ]
            )
            if all(has_targets)
            else None
        ),
    ).validate()
