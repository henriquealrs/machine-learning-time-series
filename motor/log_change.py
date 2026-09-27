"""Separate forecasting path using measured current consumption as a reference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reporting import DEFAULT_OUTPUT_DIR, evaluate_predictions, save_mse_plot
from .targets import CURRENT_CONSUMPTION
from .training import TrainingResult, train_model


def prepare_log_change_data(
    features: pd.DataFrame, targets: pd.DataFrame, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add measured t0 consumption to X and keep only future consumption in y."""
    if not (features.index.equals(targets.index) and features.index.equals(metadata.index)):
        raise ValueError("X, y, and metadata must have matching row indices.")
    future_columns = [column for column in targets if int(column.rsplit("_", 1)[1]) > 0]
    if not future_columns:
        raise ValueError("Log-change forecasting requires at least one future horizon.")
    inputs = features.copy()
    inputs[CURRENT_CONSUMPTION] = targets["fuel_consumption_t_plus_0"]
    return inputs, targets[future_columns].copy(), metadata.copy()


def train_model_log_change(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    model_name: str = "hist_gradient_boosting",
    split_type: str = "experiment",
    log_scale: float = 1.0,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> TrainingResult:
    """Train with oil temperature and evaluate reconstructed consumption."""
    inputs, future_targets, metadata = prepare_log_change_data(features, targets, metadata)
    result = train_model(
        inputs, future_targets, metadata,
        model_name=model_name, split_type=split_type, oil_policy="with_oil",
        target_mode="log_change", log_scale=log_scale, output_dir=output_dir,
    )
    test_index = result.predictions.index
    reference = inputs.loc[test_index, CURRENT_CONSUMPTION]
    persistence = pd.DataFrame(
        np.repeat(reference.to_numpy()[:, None], future_targets.shape[1], axis=1),
        index=test_index, columns=future_targets.columns,
    )
    baseline = evaluate_predictions(future_targets.loc[test_index], persistence)
    baseline.to_csv(result.output_dir / "persistence_metrics.csv", index=False)
    for metric in ("MAE", "MSE", "RMSE", "R2"):
        result.metrics[f"persistence_{metric}"] = baseline[metric].to_numpy()
    result.metrics["MSE_skill_vs_persistence"] = (
        1 - result.metrics["MSE"] / result.metrics["persistence_MSE"].replace(0, np.nan)
    )
    result.metrics.to_csv(result.output_dir / "metrics.csv", index=False)

    records = pd.read_csv(result.output_dir / "predictions.csv")
    records["reference_consumption"] = records["row_id"].map(inputs[CURRENT_CONSUMPTION])
    records["persistence_prediction"] = records["reference_consumption"]
    records.to_csv(result.output_dir / "predictions.csv", index=False)
    observed_changes = np.log1p(future_targets.loc[test_index] / log_scale).sub(
        np.log1p(reference / log_scale), axis=0
    )
    predicted_changes = pd.DataFrame(
        result.model.predict_log_change(inputs.loc[test_index]),
        index=test_index, columns=future_targets.columns,
    )
    log_records = []
    for target in future_targets:
        horizon = int(target.rsplit("_", 1)[1])
        frame = records.loc[records["horizon_seconds"].eq(horizon)].copy()
        frame["observed_log_change"] = frame["row_id"].map(observed_changes[target])
        frame["predicted_log_change"] = frame["row_id"].map(predicted_changes[target])
        log_records.append(frame)
    pd.concat(log_records, ignore_index=True).to_csv(
        result.output_dir / "log_changes.csv", index=False
    )
    save_mse_plot(result.metrics, records, result.output_dir)

    settings_path = result.output_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings.update({
        "target_transformation": "log1p(future_consumption / scale) - log1p(current_consumption / scale)",
        "reference_feature": CURRENT_CONSUMPTION,
        "reference_availability": "Measured current consumption required at prediction time",
        "reconstruction": "max(0, scale * expm1(log1p(current / scale) + predicted_log_change))",
        "baseline": "Repeat measured current consumption at every future horizon",
    })
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return result
