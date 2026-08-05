"""Tests for the 1-20 pair hierarchical component-field pipeline."""

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from physics_foundation import (
    ComponentGraphSepONet,
    MultiPairFieldService,
    compose_multi_pair_graph,
    load_component_case,
)

CASE = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"
CHECKPOINT = PROJECT / "outputs" / "component_graph_seponet_20pairs.pt"
SUMMARY = PROJECT / "outputs" / "component_graph_seponet_20pairs.json"


class MultiPairFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_component_case(CASE, 4)

    def test_composed_graph_scales_to_twenty_pairs(self):
        one, one_layout = compose_multi_pair_graph(
            self.base, 1, max_probes_per_component=4
        )
        twenty, twenty_layout = compose_multi_pair_graph(
            self.base, 20, max_probes_per_component=4
        )
        self.assertEqual(one.node_features.shape, (9, 16))
        self.assertEqual(twenty.node_features.shape, (123, 16))
        self.assertEqual(twenty_layout.rows, 4)
        self.assertEqual(twenty_layout.columns, 5)
        self.assertEqual(twenty_layout.probe_pair.max().item(), 19)
        base_values = self.base.probe_targets[:, 1][self.base.probe_target_mask[:, 1]]
        base_span = base_values.max() - base_values.min()
        one_values = one.probe_targets[:, 1][one.probe_target_mask[:, 1]]
        twenty_values = twenty.probe_targets[:, 1][twenty.probe_target_mask[:, 1]]
        one_span = one_values.max() - one_values.min()
        twenty_span = twenty_values.max() - twenty_values.min()
        self.assertTrue(
            torch.isclose(twenty_span, one_span + 19 * base_span, rtol=1e-5)
        )

    def test_one_model_accepts_one_and_twenty_pair_graphs(self):
        graphs = [
            compose_multi_pair_graph(
                self.base, count, max_probes_per_component=2
            )[0]
            for count in (1, 20)
        ]
        model = ComponentGraphSepONet(
            node_dim=16,
            edge_dim=8,
            global_dim=4,
            num_component_types=6,
            hidden_dim=12,
            rank=3,
            message_passing_steps=4,
        )
        for graph in graphs:
            output = model(graph)
            self.assertEqual(output.shape, (len(graph.probe_coords), 2))
            self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(CHECKPOINT.exists(), "20-pair checkpoint is not present")
    def test_field_service_enforces_supported_domain_and_series_voltage(self):
        service = MultiPairFieldService(CHECKPOINT, CASE, SUMMARY)
        one = service.predict(1, 0.5, 323.15)
        twenty = service.predict(20, 0.5, 323.15)
        self.assertEqual(len(twenty["pairs"]), 20)
        self.assertAlmostEqual(
            twenty["metrics"]["terminal_voltage_V"],
            20 * one["metrics"]["terminal_voltage_V"],
            places=4,
        )
        self.assertAlmostEqual(
            sum(pair["voltage_drop_V"] for pair in twenty["pairs"]),
            twenty["metrics"]["terminal_voltage_V"],
            places=4,
        )
        self.assertLess(twenty["metrics"]["inference_ms"], 2000)
        with self.assertRaisesRegex(ValueError, "between 1 and 20"):
            service.predict(21, 0.5, 323.15)


if __name__ == "__main__":
    unittest.main()
