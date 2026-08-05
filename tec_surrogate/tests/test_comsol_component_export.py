"""Tests for COMSOL component geometry conversion without starting COMSOL."""

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np

from physics_foundation import load_component_case, load_component_interfaces
from physics_foundation.comsol_export import (
    ComponentDefinition,
    _normalise_result,
    build_component_probes,
    infer_contacts,
    local_probe_grid,
)

REAL_EXPORT = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"


class ComsolComponentExportTests(unittest.TestCase):
    def test_local_probe_grid_is_cell_centred(self):
        grid = local_probe_grid((2, 3, 4))
        self.assertEqual(grid.shape, (24, 3))
        self.assertTrue(np.all(grid > 0))
        self.assertTrue(np.all(grid < 1))

    def test_axis_aligned_contacts_are_inferred(self):
        components = [
            ComponentDefinition("bottom", "ceramic", 1, (0, 1, 0, 1, 0, 0.1)),
            ComponentDefinition("leg", "p_leg", 2, (0.2, 0.4, 0.2, 0.4, 0.1, 0.9)),
            ComponentDefinition("top", "ceramic", 3, (0, 1, 0, 1, 0.9, 1.0)),
        ]
        contacts = infer_contacts(components)
        self.assertEqual([(item.source, item.target) for item in contacts], [(0, 1), (1, 2)])
        self.assertTrue(all(item.axis == 2 for item in contacts))

        local, component_ids, physical = build_component_probes(
            components,
            grids={"ceramic": (1, 1, 1), "p_leg": (1, 1, 1)},
        )
        self.assertEqual(local.shape, (3, 3))
        self.assertEqual(component_ids.tolist(), [0, 1, 2])
        self.assertTrue(np.allclose(physical[1], [0.3, 0.3, 0.5]))

    def test_comsol_result_shapes_are_normalized(self):
        result = _normalise_result(np.arange(12).reshape(1, 3, 4), point_count=4)
        self.assertEqual(result.shape, (3, 4))
        singleton = _normalise_result(np.arange(4).reshape(1, 1, 4), point_count=4)
        self.assertEqual(singleton.shape, (1, 4))

    @unittest.skipUnless(REAL_EXPORT.exists(), "real COMSOL component export is not present")
    def test_real_export_loads_as_graph(self):
        graph = load_component_case(REAL_EXPORT, solution_index=0)
        self.assertEqual(graph.node_features.shape, (7, 10))
        self.assertEqual(graph.edge_index.shape, (2, 14))
        self.assertEqual(graph.probe_targets.shape, (960, 2))
        self.assertEqual(graph.probe_target_mask[:, 0].sum().item(), 960)
        self.assertEqual(graph.probe_target_mask[:, 1].sum().item(), 864)

        interfaces = load_component_interfaces(REAL_EXPORT, samples_per_axis=2)
        self.assertEqual(len(interfaces.source_components), 28)
        self.assertEqual(interfaces.thermal_mask.sum().item(), 28)
        self.assertEqual(interfaces.electrical_mask.sum().item(), 16)
        self.assertTrue((interfaces.source_coords >= 0).all())
        self.assertTrue((interfaces.source_coords <= 1).all())
        self.assertTrue((interfaces.target_coords >= 0).all())
        self.assertTrue((interfaces.target_coords <= 1).all())


if __name__ == "__main__":
    unittest.main()
