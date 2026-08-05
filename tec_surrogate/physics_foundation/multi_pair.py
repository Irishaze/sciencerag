"""Compose variable-PN TEC graphs from the calibrated one-pair COMSOL case."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import torch
from torch.nn import functional as F

from .graph import ComponentGraph, Tensor


MULTI_COMPONENT_TYPES = {
    "ceramic": 0,
    "conductor": 1,
    "p_leg": 2,
    "n_leg": 3,
    "pair": 4,
    "module": 5,
}
MAX_SUPPORTED_PAIRS = 20

# Electrical order and one-pair COMSOL component indices.
CELL_ROLES = (
    ("conductor_in", 2),
    ("n_leg", 3),
    ("conductor_bridge", 4),
    ("p_leg", 6),
    ("conductor_out", 5),
)


@dataclass(frozen=True)
class MultiPairLayout:
    """Probe and node semantics aligned with a composed graph."""

    n_pairs: int
    rows: int
    columns: int
    node_pair: Tensor
    node_role: tuple[str, ...]
    probe_pair: Tensor
    probe_role: tuple[str, ...]


def _grid_shape(n_pairs: int) -> tuple[int, int]:
    columns = ceil(sqrt(n_pairs))
    rows = ceil(n_pairs / columns)
    return rows, columns


def _type_for_role(role: str) -> int:
    if role.startswith("conductor"):
        return MULTI_COMPONENT_TYPES["conductor"]
    return MULTI_COMPONENT_TYPES[role]


def compose_multi_pair_graph(
    base: ComponentGraph,
    n_pairs: int,
    current: float | None = None,
    max_probes_per_component: int | None = None,
) -> tuple[ComponentGraph, MultiPairLayout]:
    """Replicate the one-pair field template into a hierarchical 1-20 pair graph.

    The temperature template is shared by each repeated cell. Electric
    potential is shifted by the one-pair terminal span along the series path.
    Pair and module nodes carry no decoded fields; they shorten graph paths so
    every physical component receives module-wide context in a few layers.
    """

    base.validate()
    if not 1 <= n_pairs <= MAX_SUPPORTED_PAIRS:
        raise ValueError(f"n_pairs must be between 1 and {MAX_SUPPORTED_PAIRS}")
    if current is None:
        current = float(base.global_features[0, 0])
    if max_probes_per_component is not None and max_probes_per_component < 1:
        raise ValueError("max_probes_per_component must be positive")

    rows, columns = _grid_shape(n_pairs)
    feature_rows: list[Tensor] = []
    node_types: list[int] = []
    node_sizes: list[Tensor] = []
    thermal: list[Tensor] = []
    electrical: list[Tensor] = []
    seebeck: list[Tensor] = []
    field_masks: list[Tensor] = []
    node_pairs: list[int] = []
    node_roles: list[str] = []
    centers: list[Tensor] = []

    def add_node(
        role: str,
        component_type: int,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        pair_index: int,
        base_index: int | None,
    ) -> int:
        if pair_index >= 0:
            row = pair_index // columns
            column = pair_index % columns
            pair_position = pair_index / max(1, n_pairs - 1)
            row_position = row / max(1, rows - 1)
            column_position = column / max(1, columns - 1)
            terminal_distance = pair_position
        else:
            pair_position = row_position = column_position = terminal_distance = 0.0
        one_hot = F.one_hot(
            torch.tensor(component_type), num_classes=len(MULTI_COMPONENT_TYPES)
        ).to(dtype=base.node_features.dtype)
        center_tensor = base.node_features.new_tensor(center)
        size_tensor = base.node_features.new_tensor(size)
        positional = base.node_features.new_tensor(
            (pair_position, row_position, column_position, terminal_distance)
        )
        feature_rows.append(torch.cat((center_tensor, size_tensor, one_hot, positional)))
        node_types.append(component_type)
        centers.append(center_tensor)
        node_pairs.append(pair_index)
        node_roles.append(role)
        if base_index is None:
            node_sizes.append(base.node_sizes.new_ones(3))
            thermal.append(base.thermal_conductivity.new_tensor(1.0))
            electrical.append(base.electrical_conductivity.new_tensor(0.0))
            seebeck.append(base.seebeck_coefficient.new_tensor(0.0))
            field_masks.append(torch.tensor((False, False), dtype=torch.bool))
        else:
            node_sizes.append(base.node_sizes[base_index])
            thermal.append(base.thermal_conductivity[base_index])
            electrical.append(base.electrical_conductivity[base_index])
            seebeck.append(base.seebeck_coefficient[base_index])
            field_masks.append(base.node_field_mask[base_index])
        return len(node_types) - 1

    bottom = add_node(
        "ceramic_bottom", MULTI_COMPONENT_TYPES["ceramic"],
        (0.5, 0.5, 0.04), (1.0, 1.0, 0.08), -1, 0,
    )
    top = add_node(
        "ceramic_top", MULTI_COMPONENT_TYPES["ceramic"],
        (0.5, 0.5, 0.96), (1.0, 1.0, 0.08), -1, 1,
    )

    pair_nodes: list[int] = []
    physical_nodes: list[dict[str, int]] = []
    cell_width = 1.0 / columns
    cell_height = 1.0 / rows
    for pair_index in range(n_pairs):
        row = pair_index // columns
        column = pair_index % columns
        cx = (column + 0.5) / columns
        cy = (row + 0.5) / rows
        dx = 0.16 * cell_width
        pair_map: dict[str, int] = {}
        role_geometry = {
            "conductor_in": ((cx - dx, cy, 0.14), (0.22 * cell_width, 0.42 * cell_height, 0.04)),
            "n_leg": ((cx - dx, cy, 0.50), (0.18 * cell_width, 0.36 * cell_height, 0.68)),
            "conductor_bridge": ((cx, cy, 0.86), (0.55 * cell_width, 0.42 * cell_height, 0.04)),
            "p_leg": ((cx + dx, cy, 0.50), (0.18 * cell_width, 0.36 * cell_height, 0.68)),
            "conductor_out": ((cx + dx, cy, 0.14), (0.22 * cell_width, 0.42 * cell_height, 0.04)),
        }
        for role, base_index in CELL_ROLES:
            center, size = role_geometry[role]
            pair_map[role] = add_node(
                role, _type_for_role(role), center, size, pair_index, base_index
            )
        pair_node = add_node(
            "pair", MULTI_COMPONENT_TYPES["pair"],
            (cx, cy, 0.5), (0.75 * cell_width, 0.75 * cell_height, 0.76),
            pair_index, None,
        )
        pair_nodes.append(pair_node)
        physical_nodes.append(pair_map)

    module = add_node(
        "module", MULTI_COMPONENT_TYPES["module"],
        (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), -1, None,
    )

    edge_index: list[tuple[int, int]] = []
    edge_features: list[Tensor] = []

    def connect(source: int, target: int, thermal_edge: float, electrical_edge: float) -> None:
        delta = centers[target] - centers[source]
        prefix = base.edge_features.new_tensor(
            (thermal_edge, electrical_edge, 1.0, 0.0, 0.0)
        )
        edge_index.extend(((source, target), (target, source)))
        edge_features.extend((torch.cat((prefix, delta)), torch.cat((prefix, -delta))))

    for pair_index, pair_map in enumerate(physical_nodes):
        conductor_in = pair_map["conductor_in"]
        n_leg = pair_map["n_leg"]
        bridge = pair_map["conductor_bridge"]
        p_leg = pair_map["p_leg"]
        conductor_out = pair_map["conductor_out"]
        connect(bottom, conductor_in, 1.0, 0.0)
        connect(bottom, conductor_out, 1.0, 0.0)
        connect(top, bridge, 1.0, 0.0)
        for left, right in (
            (conductor_in, n_leg),
            (n_leg, bridge),
            (bridge, p_leg),
            (p_leg, conductor_out),
        ):
            connect(left, right, 1.0, 1.0)
        for physical in pair_map.values():
            connect(physical, pair_nodes[pair_index], 0.0, 0.0)
        connect(pair_nodes[pair_index], module, 0.0, 0.0)
        if pair_index:
            connect(
                physical_nodes[pair_index - 1]["conductor_out"],
                conductor_in,
                0.0,
                1.0,
            )
            connect(pair_nodes[pair_index - 1], pair_nodes[pair_index], 1.0, 0.0)
    connect(bottom, module, 1.0, 0.0)
    connect(top, module, 1.0, 0.0)

    probe_coords: list[Tensor] = []
    probe_components: list[Tensor] = []
    probe_targets: list[Tensor] = []
    probe_masks: list[Tensor] = []
    probe_pairs: list[Tensor] = []
    probe_roles: list[str] = []
    conductive_values = base.probe_targets[:, 1][base.probe_target_mask[:, 1]]
    potential_min = conductive_values.min()
    potential_span = conductive_values.max() - potential_min

    def copy_probes(node: int, base_index: int, pair_index: int, role: str) -> None:
        indices = torch.nonzero(base.probe_components == base_index).flatten()
        if max_probes_per_component is not None and len(indices) > max_probes_per_component:
            sample_positions = torch.linspace(
                0, len(indices) - 1, max_probes_per_component
            ).round().long()
            indices = indices[sample_positions]
        coords = base.probe_coords[indices]
        targets = base.probe_targets[indices].clone()
        masks = base.probe_target_mask[indices].clone()
        if pair_index >= 0 and torch.any(masks[:, 1]):
            targets[:, 1] = targets[:, 1] - potential_min + pair_index * potential_span
        probe_coords.append(coords)
        probe_components.append(torch.full((len(indices),), node, dtype=torch.long))
        probe_targets.append(targets)
        probe_masks.append(masks)
        probe_pairs.append(torch.full((len(indices),), pair_index, dtype=torch.long))
        probe_roles.extend([role] * len(indices))

    copy_probes(bottom, 0, -1, "ceramic_bottom")
    copy_probes(top, 1, -1, "ceramic_top")
    for pair_index, pair_map in enumerate(physical_nodes):
        for role, base_index in CELL_ROLES:
            copy_probes(pair_map[role], base_index, pair_index, role)

    global_features = base.global_features.new_tensor(
        [[current, n_pairs / MAX_SUPPORTED_PAIRS, float(base.global_features[0, 1]) / 350.0,
          float(base.global_features[0, 2]) / 80.0]]
    )
    graph = ComponentGraph(
        node_features=torch.stack(feature_rows),
        node_types=torch.tensor(node_types, dtype=torch.long),
        node_field_mask=torch.stack(field_masks),
        node_sizes=torch.stack(node_sizes),
        thermal_conductivity=torch.stack(thermal),
        electrical_conductivity=torch.stack(electrical),
        seebeck_coefficient=torch.stack(seebeck),
        edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
        edge_features=torch.stack(edge_features),
        global_features=global_features,
        node_graph=torch.zeros(len(node_types), dtype=torch.long),
        probe_coords=torch.cat(probe_coords),
        probe_components=torch.cat(probe_components),
        probe_targets=torch.cat(probe_targets),
        probe_target_mask=torch.cat(probe_masks),
    ).validate()
    layout = MultiPairLayout(
        n_pairs=n_pairs,
        rows=rows,
        columns=columns,
        node_pair=torch.tensor(node_pairs, dtype=torch.long),
        node_role=tuple(node_roles),
        probe_pair=torch.cat(probe_pairs),
        probe_role=tuple(probe_roles),
    )
    return graph, layout
