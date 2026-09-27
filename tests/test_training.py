"""Checks for experiment isolation and complete, reusable training records."""

import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from motor.data_init import split_data
from motor.model_no_oil_temp import train_model_ignore_oil
from motor.training import train_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            "vehicle_speed": range(9),
            "oil_temperature": [350.0] * 7 + [np.nan] * 2,
            "pi_7": [1.0] * 7 + [np.nan] * 2,
        })
        self.targets = pd.DataFrame({
            "fuel_consumption_t_plus_0": np.arange(9, dtype=float),
            "fuel_consumption_t_plus_1": np.arange(9, dtype=float) + 1,
        })
        self.metadata = pd.DataFrame({
            "experiment": ["D1T1A"] * 4 + ["D1T1B"] * 3 + ["D2T1"] * 2,
            "sample": [0, 1, 2, 3, 0, 1, 2, 0, 1],
            "time_seconds": [0, 1, 2, 3, 0, 1, 2, 0, 1],
        })

    def test_split_ignores_absent_experiments(self):
        X_train, _, X_test, _ = split_data(self.features, self.targets, self.metadata)
        self.assertEqual(len(X_train), 7)
        self.assertEqual(self.metadata.loc[X_test.index, "experiment"].unique().tolist(), ["D2T1"])
        with self.assertRaisesRegex(ValueError, "Unknown split"):
            split_data(self.features, self.targets, self.metadata, "unsupported")

    def test_saved_runs_preserve_records_and_do_not_overwrite(self):
        before = self.features.copy(deep=True)
        with tempfile.TemporaryDirectory() as output, patch(
            "motor.training.build_model", return_value=DummyRegressor()
        ):
            ignore = train_model_ignore_oil(
                self.features, self.targets, self.metadata, output_dir=output
            )
            repeated = train_model_ignore_oil(
                self.features, self.targets, self.metadata, output_dir=output
            )
            self.assertNotEqual(ignore.output_dir, repeated.output_dir)
            self.assertEqual(ignore.output_dir.parent.name, "ignore_oil")
            self.assertEqual(ignore.output_dir.parent.parent.name, "experiment")
            settings = json.loads((ignore.output_dir / "settings.json").read_text())
            self.assertEqual(settings["features"], ["vehicle_speed"])
            self.assertEqual(settings["test_experiments"], ["D2T1"])
            records = pd.read_csv(ignore.output_dir / "predictions.csv")
            self.assertEqual(len(records), 4)
            np.testing.assert_allclose(
                records["target_time_seconds"],
                records["origin_time_seconds"] + records["horizon_seconds"],
            )
            self.assertTrue((ignore.output_dir / "plots/D2T1.png").is_file())
            self.assertTrue((ignore.output_dir / "plots/D2T1_residuals.png").is_file())
            np.testing.assert_allclose(records["residual"], [4, 5, 4, 5])
            self.assertTrue((ignore.output_dir / "plots/mse_by_horizon.png").is_file())
            np.testing.assert_allclose(ignore.metrics["MSE"], [20.5, 20.5])
            np.testing.assert_allclose(ignore.metrics["MSE"], ignore.metrics["RMSE"].pow(2))
            np.testing.assert_allclose(records["squared_error"], records["residual"].pow(2))
            saved_model = joblib.load(ignore.output_dir / "model.joblib")
            np.testing.assert_allclose(
                saved_model.predict(self.features.loc[ignore.predictions.index, ["vehicle_speed"]]),
                ignore.predictions,
            )
            with_oil = train_model(
                self.features, self.targets, self.metadata,
                oil_policy="with_oil", output_dir=output,
            )
            settings = json.loads((with_oil.output_dir / "settings.json").read_text())
            self.assertEqual(settings["test_experiments"], ["D1T1B"])
            self.assertEqual(settings["oil_policy_excluded_samples"], 2)
            split = pd.read_csv(with_oil.output_dir / "split.csv")
            self.assertEqual(split["partition"].value_counts()["excluded"], 2)
            self.assertEqual(with_oil.output_dir.parent.name, "with_oil")
            self.assertTrue((with_oil.output_dir / "plots/D1T1B_residuals.png").is_file())
        pd.testing.assert_frame_equal(before, self.features)


if __name__ == "__main__":
    unittest.main()
