"""Tests for the variable-topology component graph SepONet prototype."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from physics_foundation import (
    ComponentGraphSepONet,
    LossScales,
    batch_component_graphs,
    interface_conservation_loss,
    steady_thermoelectric_residual_loss,
    supervised_field_loss,
)
from physics_foundation.synthetic import COMPONENT_TYPES, build_synthetic_tec_graph


class PhysicsFoundationTests(unittest.TestCase):
    def _model(self, graph):
        return ComponentGraphSepONet(
            node_dim=graph.node_features.shape[1],
            edge_dim=graph.edge_features.shape[1],
            global_dim=graph.global_features.shape[1],
            num_component_types=len(COMPONENT_TYPES),
            hidden_dim=16,
            rank=4,
            message_passing_steps=1,
        )

    def test_one_model_accepts_different_pair_counts(self):
        graph_one, _, _ = build_synthetic_tec_graph(1)
        graph_three, _, _ = build_synthetic_tec_graph(3)
        model = self._model(graph_one)
        output_one = model(graph_one)
        output_three = model(graph_three)
        self.assertEqual(output_one.shape, (len(graph_one.probe_coords), 2))
        self.assertEqual(output_three.shape, (len(graph_three.probe_coords), 2))
        self.assertTrue(torch.isfinite(output_three).all())

    def test_variable_graph_batch_offsets_indices(self):
        graphs = [build_synthetic_tec_graph(count)[0] for count in (1, 2, 4)]
        batch = batch_component_graphs(graphs)
        model = self._model(batch)
        output = model(batch)
        self.assertEqual(len(batch.global_features), 3)
        self.assertEqual(len(output), sum(len(graph.probe_coords) for graph in graphs))
        self.assertLess(int(batch.edge_index.max()), len(batch.node_features))

    def test_data_interface_and_pde_losses_backpropagate(self):
        graph, interfaces, _ = build_synthetic_tec_graph(1)
        model = self._model(graph)
        node_state = model.encode_nodes(graph)
        prediction = model.decode(
            graph, node_state, graph.probe_components, graph.probe_coords
        )
        scales = LossScales(
            temperature=50.0,
            potential=5.0,
            heat_flux=10.0,
            current_density=1e5,
            charge_residual=1e5,
            energy_residual=1e5,
        )
        data_loss = supervised_field_loss(
            prediction, graph.probe_targets, scales, graph.probe_target_mask
        )
        interface_loss, _ = interface_conservation_loss(
            model, graph, node_state, interfaces, scales
        )
        pde_loss, _ = steady_thermoelectric_residual_loss(
            model,
            graph,
            node_state,
            component_ids=torch.tensor([2, 3]),
            local_coords=torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.5, 0.4]]),
            scales=scales,
        )
        total = data_loss + 1e-4 * interface_loss + 1e-8 * pde_loss
        total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_interface_masks_disable_inapplicable_electrical_terms(self):
        graph, interfaces, _ = build_synthetic_tec_graph(1)
        interfaces = replace(
            interfaces,
            electrical_mask=torch.zeros(
                len(interfaces.source_components), dtype=torch.bool
            ),
        )
        model = self._model(graph)
        node_state = model.encode_nodes(graph)
        _, terms = interface_conservation_loss(
            model, graph, node_state, interfaces, LossScales()
        )
        self.assertEqual(terms["potential_jump"].item(), 0.0)
        self.assertEqual(terms["current_balance"].item(), 0.0)
        self.assertGreaterEqual(terms["temperature_jump"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
