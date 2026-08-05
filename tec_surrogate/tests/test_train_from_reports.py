"""Tests for semantic COMSOL report parsing and latent prediction."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location(
    "train_from_reports", PROJECT / "scripts" / "09_train_from_reports.py"
)
REPORTS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORTS)


class ReportTrainingTests(unittest.TestCase):
    def test_unicode_scientific_notation(self):
        self.assertAlmostEqual(REPORTS.parse_comsol_number("1.0776E−4 m"), 1.0776e-4)

    def test_sample_29_semantic_parse(self):
        sample = REPORTS.parse_report(PROJECT / "data" / "raw" / "doc" / "sample_29.docx")
        self.assertAlmostEqual(sample["inputs"]["length_mm"], 5.7253)
        self.assertAlmostEqual(sample["inputs"]["d_conductor_um"], 107.76)
        self.assertEqual(sample["inputs"]["n_pairs"], 2.0)
        self.assertEqual(sample["cop_surface"].shape, (3, 8))
        self.assertAlmostEqual(sample["scalars"]["total_resistance_ohm"], 0.086238)

    def test_dataset_deduplication_and_prediction_shape(self):
        dataset = REPORTS.extract_dataset(PROJECT / "data" / "raw" / "doc")
        self.assertEqual(dataset["source_file_count"], 35)
        self.assertEqual(len(dataset["filenames"]), 34)
        self.assertEqual(dataset["X"].shape, (34, 10))
        self.assertEqual(dataset["Y"].shape, (34, 30))
        self.assertEqual(len(dataset["duplicates"]), 1)

        model, _, _ = REPORTS.train_latent_surrogate(dataset)
        prediction = REPORTS.predict(model, dataset["X"][:2])
        self.assertEqual(prediction["scalar_outputs"].shape, (2, 6))
        self.assertEqual(prediction["cop_surfaces"].shape, (2, 3, 8))
        self.assertTrue(np.all(np.isfinite(prediction["output_vector"])))


if __name__ == "__main__":
    unittest.main()
