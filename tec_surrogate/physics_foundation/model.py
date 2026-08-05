"""Component graph processor with per-component separable field decoders."""

from __future__ import annotations

import math

import torch
from torch import nn

from .graph import ComponentGraph, Tensor


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, depth - 1)):
        layers.extend((nn.Linear(current, hidden_dim), activation()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class MessagePassingLayer(nn.Module):
    """Directed edge message passing implemented without external graph packages."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message = _mlp(2 * hidden_dim + edge_dim, hidden_dim, hidden_dim, depth=3)
        self.update = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, depth=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        if edge_index.shape[1] == 0:
            return node_state
        source, target = edge_index
        messages = self.message(
            torch.cat((node_state[source], node_state[target], edge_features), dim=-1)
        )
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(len(node_state), device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, target, torch.ones(len(target), device=node_state.device, dtype=node_state.dtype))
        aggregate = aggregate / degree.clamp_min(1).unsqueeze(-1)
        return self.norm(node_state + self.update(torch.cat((node_state, aggregate), dim=-1)))


class ComponentGraphSepONet(nn.Module):
    """Predict temperature and electric potential on variable component graphs.

    The graph processor computes one context vector per physical component. A
    separable decoder then evaluates continuous local fields using independent
    trunk networks for xi, eta, and zeta. Trunks are shared by every component
    with the same type, so changing the number of PN pairs does not change the
    number of trainable parameters.
    """

    field_names = ("temperature", "potential")

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        num_component_types: int,
        hidden_dim: int = 64,
        rank: int = 32,
        message_passing_steps: int = 4,
        type_embedding_dim: int = 8,
        trunk_depth: int = 3,
    ) -> None:
        super().__init__()
        if num_component_types < 1 or rank < 1:
            raise ValueError("num_component_types and rank must be positive")
        self.num_component_types = num_component_types
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_fields = len(self.field_names)

        self.type_embedding = nn.Embedding(num_component_types, type_embedding_dim)
        self.node_encoder = _mlp(
            node_dim + 3 + global_dim + type_embedding_dim,
            hidden_dim,
            hidden_dim,
            depth=3,
        )
        self.graph_layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim, edge_dim) for _ in range(message_passing_steps)
        )
        self.branch = _mlp(
            hidden_dim,
            self.num_fields * rank,
            hidden_dim,
            depth=3,
        )
        self.field_bias = nn.Linear(hidden_dim, self.num_fields)

        self.trunks = nn.ModuleList(
            nn.ModuleList(
                _mlp(
                    1,
                    self.num_fields * rank,
                    hidden_dim,
                    depth=trunk_depth,
                    activation=nn.Tanh,
                )
                for _ in range(num_component_types)
            )
            for _ in range(3)
        )

    def encode_nodes(self, graph: ComponentGraph) -> Tensor:
        graph.validate()
        if torch.any(graph.node_types < 0) or torch.any(graph.node_types >= self.num_component_types):
            raise ValueError("node_types contains a component type not configured in the model")
        graph_context = graph.global_features[graph.node_graph]
        material_context = torch.stack(
            (
                torch.log1p(graph.thermal_conductivity.clamp_min(0.0)) / 10.0,
                torch.log1p(graph.electrical_conductivity.clamp_min(0.0)) / 20.0,
                graph.seebeck_coefficient * 1.0e4,
            ),
            dim=-1,
        )
        state = self.node_encoder(
            torch.cat(
                (
                    graph.node_features,
                    material_context,
                    self.type_embedding(graph.node_types),
                    graph_context,
                ),
                dim=-1,
            )
        )
        for layer in self.graph_layers:
            state = layer(state, graph.edge_index, graph.edge_features)
        return state

    def decode(
        self,
        graph: ComponentGraph,
        node_state: Tensor,
        component_ids: Tensor,
        local_coords: Tensor,
    ) -> Tensor:
        if local_coords.ndim != 2 or local_coords.shape[1] != 3:
            raise ValueError("local_coords must have shape (P, 3)")
        if component_ids.shape != (len(local_coords),):
            raise ValueError("component_ids must have shape (P,)")

        selected_state = node_state[component_ids]
        branch = self.branch(selected_state).reshape(-1, self.num_fields, self.rank)
        basis = torch.ones_like(branch)
        selected_types = graph.node_types[component_ids]

        for axis in range(3):
            axis_basis = torch.zeros_like(branch)
            for component_type, trunk in enumerate(self.trunks[axis]):
                mask = selected_types == component_type
                if torch.any(mask):
                    values = trunk(local_coords[mask, axis : axis + 1])
                    axis_basis[mask] = values.reshape(-1, self.num_fields, self.rank)
            basis = basis * axis_basis

        decoded = (branch * basis).sum(dim=-1) / math.sqrt(self.rank)
        return self.field_bias(selected_state) + decoded

    def forward(self, graph: ComponentGraph) -> Tensor:
        node_state = self.encode_nodes(graph)
        return self.decode(graph, node_state, graph.probe_components, graph.probe_coords)
