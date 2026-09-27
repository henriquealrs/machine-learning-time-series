"""JSON configuration, training-only preprocessing, and saved MLP predictions."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from motor.neural_network_config import load_neural_network_config
from motor.training import train_model


class NeuralNetworkTests(unittest.TestCase):
    def test_alternative_config_overrides_defaults_and_rejects_errors(self):
        default = load_neural_network_config()
        self.assertEqual(default.hidden_layers, [30, 10])
        self.assertEqual(default.function, "relu")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"hidden_layers": [8], "function": "tanh"}))
            alternative = load_neural_network_config(path)
            self.assertEqual(alternative.hidden_layers, [8])
            self.assertEqual(alternative.function, "tanh")
            self.assertEqual(alternative.max_iter, default.max_iter)
            for invalid in (
                {"hidden_layers": [0]}, {"function": "unknown"},
                {"max_iter": 0}, {"learning_rate_init": "fast"},
                {"typo": 1}, {"batch_size": True},
            ):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_neural_network_config(path)

    def test_preprocessing_uses_train_rows_and_saved_model_reconstructs_targets(self):
        X = pd.DataFrame({
            "vehicle_speed": [0., 1., 2., 3., 4., 5., 6., 7., 1000., 1001.],
            "engine_torque": [0., np.nan, 2., 3., 4., 5., 6., 7., np.nan, 1001.],
            "oil_temperature": [350.] * 10,
            "pi_7": [1.] * 10,
        })
        y = pd.DataFrame({
            "fuel_consumption_t_plus_0": np.arange(10, dtype=float) * 100,
            "fuel_consumption_t_plus_1": np.arange(10, dtype=float) * 100 + 20,
        })
        metadata = pd.DataFrame({
            "experiment": ["A"] * 8 + ["B"] * 2,
            "sample": list(range(8)) + [0, 1],
            "time_seconds": list(range(8)) + [0, 1],
        })
        config = asdict(load_neural_network_config())
        config.update(hidden_layers=[4], solver="lbfgs", max_iter=30)
        with tempfile.TemporaryDirectory() as folder:
            result = train_model(
                X, y, metadata, model_name="neural_network",
                model_parameters=config, oil_policy="with_oil", output_dir=folder,
            )
            pipeline = result.model.regressor_
            self.assertEqual(pipeline.named_steps["network"].hidden_layer_sizes, (4,))
            self.assertFalse(pipeline.named_steps["network"].early_stopping)
            self.assertAlmostEqual(pipeline.named_steps["scaler"].mean_[0], 3.5)
            self.assertAlmostEqual(pipeline.named_steps["imputer"].statistics_[1], 4.0)
            np.testing.assert_allclose(result.model.transformer_.mean_, y.iloc[:8].mean())
            saved = joblib.load(result.output_dir / "model.joblib")
            np.testing.assert_allclose(saved.predict(X.loc[result.predictions.index]), result.predictions)
            settings = json.loads((result.output_dir / "settings.json").read_text())
            self.assertEqual(settings["model_configuration"]["hidden_layers"], [4])
            self.assertEqual(result.output_dir.parents[3].name, "neural_network")
            self.assertTrue((result.output_dir / "plots/B_residuals.png").exists())
            self.assertTrue((result.output_dir / "plots/mse_by_horizon.png").exists())


if __name__ == "__main__":
    unittest.main()
