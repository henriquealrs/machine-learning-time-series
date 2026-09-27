"""Verify zero-safe log targets, reconstruction, and forecast-only evaluation."""

import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from motor.log_change import prepare_log_change_data, train_model_log_change
from motor.targets import CURRENT_CONSUMPTION, LogChangeRegressor


class LogChangeTests(unittest.TestCase):
    def test_transform_and_reconstruction_include_zero_consumption(self):
        X = pd.DataFrame({CURRENT_CONSUMPTION: [0.0, 2.0, 3.0]})
        y = pd.DataFrame({"future": [0.0, 4.0, 6.0]})
        model = LogChangeRegressor(DummyRegressor(), scale=2.0).fit(X, y)
        changes = np.log1p(y.to_numpy() / 2) - np.log1p(X.to_numpy() / 2)
        np.testing.assert_allclose(model.predict_log_change(X), np.repeat(changes.mean(axis=0)[None], 3, axis=0))
        expected = 2 * np.expm1(np.log1p(X.to_numpy() / 2) + changes.mean(axis=0))
        np.testing.assert_allclose(model.predict(X), expected)
        negative = LogChangeRegressor(DummyRegressor(strategy="constant", constant=-100.0)).fit(X, y)
        np.testing.assert_allclose(negative.predict(X), 0)
        with self.assertRaises(ValueError):
            LogChangeRegressor(DummyRegressor(), scale=0).fit(X, y)

    def test_separate_path_keeps_t0_as_reference_and_records_baseline(self):
        X = pd.DataFrame({
            "vehicle_speed": range(7),
            "oil_temperature": [350.0] * 5 + [np.nan] * 2,
            "pi_7": [1.0] * 5 + [np.nan] * 2,
        })
        y = pd.DataFrame({
            "fuel_consumption_t_plus_0": [0., 1., 2., 3., 4., 5., 6.],
            "fuel_consumption_t_plus_1": [1., 2., 3., 4., 5., 6., 7.],
            "fuel_consumption_t_plus_2": [2., 3., 4., 5., 6., 7., 8.],
        })
        metadata = pd.DataFrame({
            "experiment": ["A"] * 3 + ["B"] * 2 + ["C"] * 2,
            "sample": [0, 1, 2, 0, 1, 0, 1],
            "time_seconds": [0, 1, 2, 0, 1, 0, 1],
        })
        inputs, futures, _ = prepare_log_change_data(X, y, metadata)
        pd.testing.assert_series_equal(inputs[CURRENT_CONSUMPTION], y.iloc[:, 0], check_names=False)
        self.assertNotIn(y.columns[0], futures)
        self.assertNotIn(CURRENT_CONSUMPTION, X)
        with self.assertRaises(ValueError):
            prepare_log_change_data(X, y.iloc[:, :1], metadata)
        estimator = LogChangeRegressor(DummyRegressor(strategy="constant", constant=np.zeros(2)))
        with tempfile.TemporaryDirectory() as output, patch(
            "motor.training.build_model", return_value=estimator
        ):
            result = train_model_log_change(X, y, metadata, output_dir=output)
            self.assertEqual(result.output_dir.parent.name, "log_change")
            self.assertEqual(result.output_dir.parent.parent.name, "with_oil")
            np.testing.assert_allclose(result.metrics["MSE"], [1., 4.])
            np.testing.assert_allclose(result.metrics["MSE_skill_vs_persistence"], 0, atol=1e-12)
            self.assertTrue((result.output_dir / "log_changes.csv").is_file())
            settings = json.loads((result.output_dir / "settings.json").read_text())
            self.assertEqual(settings["test_experiments"], ["B"])
            self.assertNotIn(y.columns[0], settings["targets"])
            saved = joblib.load(result.output_dir / "model.joblib")
            np.testing.assert_allclose(saved.predict(inputs.loc[result.predictions.index]), result.predictions)


if __name__ == "__main__":
    unittest.main()
