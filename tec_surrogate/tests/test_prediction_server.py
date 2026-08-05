"""Tests for the prediction service used by the web UI."""

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np

from prediction_server import PredictionService


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = PredictionService()
        cls.meta = cls.service.metadata()

    def test_metadata_matches_trained_model(self):
        self.assertEqual(self.meta["model"]["sample_count"], len(self.service.X))
        self.assertEqual(self.meta["model"]["latent_dim"], 5)
        self.assertEqual(len(self.meta["inputs"]), 10)
        self.assertEqual(len(self.meta["training_latent"]), len(self.service.X))

    def test_default_prediction_is_complete(self):
        inputs = {item["name"]: item["default"] for item in self.meta["inputs"]}
        result = self.service.predict({"inputs": inputs})
        self.assertEqual(len(result["scalars"]), 6)
        self.assertEqual(np.asarray(result["cop_surface"]).shape, (3, 8))
        self.assertEqual(len(result["latent"]), 5)
        self.assertEqual(result["outside_training_range"], [])

    def test_out_of_range_input_is_reported(self):
        inputs = {item["name"]: item["default"] for item in self.meta["inputs"]}
        inputs["Tref_K"] = 500
        result = self.service.predict({"inputs": inputs})
        self.assertEqual(result["outside_training_range"][0]["name"], "Tref_K")

    def test_pair_count_must_be_integer(self):
        inputs = {item["name"]: item["default"] for item in self.meta["inputs"]}
        inputs["n_pairs"] = 2.5
        with self.assertRaisesRegex(ValueError, "integer"):
            self.service.predict({"inputs": inputs})


if __name__ == "__main__":
    unittest.main()
