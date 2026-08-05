"""Tests for the next COMSOL training batch design."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location(
    "generate_training_batch", PROJECT / "scripts" / "10_generate_comsol_training_batch.py"
)
GENERATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATOR)


class TrainingBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.summary = GENERATOR.generate_batch()

    def test_count_ids_and_groups(self):
        self.assertEqual(len(self.rows), 50)
        self.assertEqual([row["sample_id"] for row in self.rows], list(range(100, 150)))
        self.assertEqual(sum(row["batch_group"] == "core_40" for row in self.rows), 40)
        self.assertEqual(sum(row["batch_group"] == "extension_10" for row in self.rows), 10)

    def test_all_points_are_feasible_and_unique(self):
        vectors = []
        for row in self.rows:
            vector = np.array([row[name] for name in GENERATOR.PARAMETER_NAMES], dtype=float)
            metadata = GENERATOR.geometry_metadata(vector)
            self.assertIsNotNone(metadata)
            self.assertGreaterEqual(metadata["expected_n_pairs"], 1)
            self.assertLessEqual(metadata["expected_n_pairs"], 20)
            vectors.append(tuple(vector))
        self.assertEqual(len(set(vectors)), 50)

    def test_current_batch_runner_columns_are_present(self):
        required = {
            "sample_id",
            "length_mm",
            "width_mm",
            "height_mm",
            "leg_length_mm",
            "leg_width_mm",
            "pitch_mm",
            "d_conductor_um",
            "d_ceramics_mm",
            "Tref_K",
        }
        self.assertTrue(required.issubset(self.rows[0]))

    def test_rare_five_pair_region_is_covered(self):
        five_pair_rows = [row for row in self.rows if row["expected_n_pairs"] == 5]
        self.assertGreaterEqual(len(five_pair_rows), 2)

    def test_high_pair_regions_are_covered(self):
        counts = [row["expected_n_pairs"] for row in self.rows]
        for n_pairs in (8, 12, 16, 20):
            self.assertGreaterEqual(counts.count(n_pairs), 2)

    def test_design_is_not_too_close_to_existing_reports(self):
        self.assertGreater(self.summary["normalized_distance"]["minimum_selected_to_existing"], 0.25)
        self.assertGreater(self.summary["normalized_distance"]["minimum_between_selected"], 0.25)


if __name__ == "__main__":
    unittest.main()
