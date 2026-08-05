"""Small synthetic TEC graphs for smoke tests and architecture prototyping."""

from __future__ import annotations

import itertools

import torch

from .graph import BoundarySamples, ComponentGraph, InterfaceSamples


COMPONENT_TYPES = {
    "ceramic": 0,
    "conductor": 1,
    "p_leg": 2,
    "n_leg": 3,
}


def _grid(resolution: int) -> torch.Tensor:
    axis = (torch.arange(resolution, dtype=torch.float32) + 0.5) / resolution
    return torch.cartesian_prod(axis, axis, axis)


def build_synthetic_tec_graph(
    n_pairs: int,
    probe_resolution: int = 2,
    current: float = 1.5,
    cold_temperature: float = 295.0,
    hot_temperature: float = 325.0,
) -> tuple[ComponentGraph, InterfaceSamples, BoundarySamples]:
    """Build a simple variable-size graph with analytic field targets.

    It is intentionally not a replacement for COMSOL data. Its purpose is to
    exercise graph construction, shared component decoders, and gradients.
    """

    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")

    node_types = [COMPONENT_TYPES["ceramic"], COMPONENT_TYPES["ceramic"]]
    centers = [[0.5, 0.5, 0.05], [0.5, 0.5, 0.95]]
    sizes = [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]
    material = [(1.5, 1e-8, 0.0), (1.5, 1e-8, 0.0)]

    for pair in range(n_pairs):
        pair_center = (pair + 0.5) / n_pairs
        for offset, component_type, seebeck in (
            (-0.12 / n_pairs, COMPONENT_TYPES["p_leg"], 2e-4),
            (0.12 / n_pairs, COMPONENT_TYPES["n_leg"], -2e-4),
        ):
            node_types.append(component_type)
            centers.append([pair_center + offset, 0.5, 0.5])
            sizes.append([0.15 / n_pairs, 0.2, 0.8])
            material.append((1.4, 1.0e5, seebeck))

    node_types_tensor = torch.tensor(node_types, dtype=torch.long)
    centers_tensor = torch.tensor(centers, dtype=torch.float32)
    sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
    material_tensor = torch.tensor(material, dtype=torch.float32)
    type_one_hot = torch.nn.functional.one_hot(
        node_types_tensor, num_classes=len(COMPONENT_TYPES)
    ).float()
    node_features = torch.cat((centers_tensor, sizes_tensor, type_one_hot), dim=-1)

    directed_edges: list[tuple[int, int]] = []
    directed_features: list[list[float]] = []
    source_components: list[int] = []
    target_components: list[int] = []
    source_coords: list[list[float]] = []
    target_coords: list[list[float]] = []
    normals: list[list[float]] = []

    for leg in range(2, len(node_types)):
        for source, target, thermal, electrical in (
            (0, leg, 1.0, 0.0),
            (leg, 1, 1.0, 0.0),
        ):
            delta = centers_tensor[target] - centers_tensor[source]
            edge_feature = [thermal, electrical, 1.0, 0.0, 0.0, *delta.tolist()]
            reverse_feature = [thermal, electrical, 1.0, 0.0, 0.0, *(-delta).tolist()]
            directed_edges.extend(((source, target), (target, source)))
            directed_features.extend((edge_feature, reverse_feature))

            if source == 0:
                source_coords.append([centers_tensor[leg, 0], 0.5, 1.0])
                target_coords.append([0.5, 0.5, 0.0])
            else:
                source_coords.append([0.5, 0.5, 1.0])
                target_coords.append([centers_tensor[leg, 0], 0.5, 0.0])
            source_components.append(source)
            target_components.append(target)
            normals.append([0.0, 0.0, 1.0])

    # Add an electrical series path through all legs.
    legs = list(range(2, len(node_types)))
    for left, right in itertools.pairwise(legs):
        delta = centers_tensor[right] - centers_tensor[left]
        feature = [0.0, 1.0, 1.0, 0.0, 0.0, *delta.tolist()]
        reverse = [0.0, 1.0, 1.0, 0.0, 0.0, *(-delta).tolist()]
        directed_edges.extend(((left, right), (right, left)))
        directed_features.extend((feature, reverse))

    local_grid = _grid(probe_resolution)
    probe_coords = local_grid.repeat(len(node_types), 1)
    probe_components = torch.arange(len(node_types)).repeat_interleave(len(local_grid))
    selected_centers = centers_tensor[probe_components]
    selected_sizes = sizes_tensor[probe_components]
    physical_z = selected_centers[:, 2] + (probe_coords[:, 2] - 0.5) * selected_sizes[:, 2]
    temperature = cold_temperature + (hot_temperature - cold_temperature) * physical_z
    potential = current * (selected_centers[:, 0] + 0.1 * probe_coords[:, 2])
    probe_targets = torch.stack((temperature, potential), dim=-1)

    graph = ComponentGraph(
        node_features=node_features,
        node_types=node_types_tensor,
        node_field_mask=torch.ones((len(node_types), 2), dtype=torch.bool),
        node_sizes=sizes_tensor,
        thermal_conductivity=material_tensor[:, 0],
        electrical_conductivity=material_tensor[:, 1],
        seebeck_coefficient=material_tensor[:, 2],
        edge_index=torch.tensor(directed_edges, dtype=torch.long).T.contiguous(),
        edge_features=torch.tensor(directed_features, dtype=torch.float32),
        global_features=torch.tensor(
            [[current, cold_temperature, hot_temperature]], dtype=torch.float32
        ),
        node_graph=torch.zeros(len(node_types), dtype=torch.long),
        probe_coords=probe_coords,
        probe_components=probe_components,
        probe_targets=probe_targets,
        probe_target_mask=torch.ones_like(probe_targets, dtype=torch.bool),
    ).validate()

    interface_count = len(source_components)
    interfaces = InterfaceSamples(
        source_components=torch.tensor(source_components, dtype=torch.long),
        target_components=torch.tensor(target_components, dtype=torch.long),
        source_coords=torch.tensor(source_coords, dtype=torch.float32),
        target_coords=torch.tensor(target_coords, dtype=torch.float32),
        source_normals=torch.tensor(normals, dtype=torch.float32),
        thermal_resistance=torch.zeros(interface_count),
        electrical_resistance=torch.zeros(interface_count),
    ).validate()

    boundary_coords = torch.tensor(
        [[0.25, 0.25, 0.0], [0.75, 0.75, 1.0]], dtype=torch.float32
    )
    boundaries = BoundarySamples(
        components=torch.tensor([0, 1], dtype=torch.long),
        coords=boundary_coords,
        targets=torch.tensor(
            [[cold_temperature, 0.0], [hot_temperature, 0.0]], dtype=torch.float32
        ),
        mask=torch.tensor([[True, False], [True, False]]),
    ).validate()
    return graph, interfaces, boundaries
