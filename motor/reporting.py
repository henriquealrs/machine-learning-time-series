"""Evaluate predictions and persist reproducible experiment artifacts."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def create_run_directory(output_dir: Path, model_name: str, split_type: str, oil_policy: str) -> Path:
    """Preserve each execution in a new timestamped directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_dir / model_name / split_type / oil_policy / f"{timestamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def evaluate_predictions(targets: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute consumption-scale metrics separately for each target horizon."""
    return pd.DataFrame({
        "target": targets.columns,
        "horizon_seconds": [int(column.rsplit("_", 1)[1]) for column in targets],
        "MAE": mean_absolute_error(targets, predictions, multioutput="raw_values"),
        "MSE": mean_squared_error(targets, predictions, multioutput="raw_values"),
        "RMSE": root_mean_squared_error(targets, predictions, multioutput="raw_values"),
        "R2": r2_score(targets, predictions, multioutput="raw_values")
        if len(targets) >= 2 else np.full(targets.shape[1], np.nan),
    })


def prediction_records(
    targets: pd.DataFrame, predictions: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Store long-form observations with both forecast-origin and target time."""
    records = []
    for target in targets:
        horizon = int(target.rsplit("_", 1)[1])
        frame = metadata.loc[targets.index].copy()
        frame.insert(0, "row_id", frame.index)
        frame = frame.rename(columns={"time_seconds": "origin_time_seconds"})
        frame["target_time_seconds"] = frame["origin_time_seconds"] + horizon
        frame["horizon_seconds"] = horizon
        frame["observed"] = targets[target]
        frame["predicted"] = predictions[target]
        frame["residual"] = frame["observed"] - frame["predicted"]
        frame["squared_error"] = frame["residual"].pow(2)
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def save_prediction_plots(records: pd.DataFrame, run_dir: Path) -> None:
    """Plot observed and predicted consumption separately for every test run."""
    plot_dir = run_dir / "plots"
    plot_dir.mkdir()
    for experiment, series in records.groupby("experiment", sort=False):
        horizons = sorted(series["horizon_seconds"].unique())
        figure = Figure(figsize=(12, 2.8 * len(horizons)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(horizons), 1, squeeze=False).ravel()
        for axis, horizon in zip(axes, horizons, strict=True):
            points = series.loc[series["horizon_seconds"].eq(horizon)].sort_values("sample")
            # Break lines if filtering/splitting left a gap in the original series.
            segments = points["sample"].diff().ne(1).cumsum()
            for segment_number, (_, segment) in enumerate(points.groupby(segments)):
                axis.plot(
                    segment["target_time_seconds"], segment["observed"],
                    label="Observed" if segment_number == 0 else None,
                    color="tab:blue", linewidth=1, alpha=0.8,
                )
                axis.plot(
                    segment["target_time_seconds"], segment["predicted"],
                    label="Predicted" if segment_number == 0 else None,
                    color="tab:orange", linewidth=1, alpha=0.8,
                )
            axis.set_title(f"{experiment} — horizon t+{horizon} s")
            axis.set_ylabel("Fuel consumption\n(source units)")
            axis.grid(alpha=0.25)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel("Target time within experiment (seconds)")
        figure.savefig(plot_dir / f"{experiment}.png", dpi=150)
        figure.clear()


def save_residual_plots(records: pd.DataFrame, run_dir: Path) -> None:
    """Plot signed instantaneous error with a zero reference at each horizon."""
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for experiment, series in records.groupby("experiment", sort=False):
        horizons = sorted(series["horizon_seconds"].unique())
        figure = Figure(figsize=(12, 2.8 * len(horizons)), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(len(horizons), 1, squeeze=False).ravel()
        limit = max(float(series["residual"].abs().max()) * 1.05, 1.0)
        for axis, horizon in zip(axes, horizons, strict=True):
            points = series.loc[series["horizon_seconds"].eq(horizon)].sort_values("sample")
            segments = points["sample"].diff().ne(1).cumsum()
            for _, segment in points.groupby(segments):
                axis.plot(
                    segment["target_time_seconds"], segment["residual"],
                    color="tab:red", linewidth=1, alpha=0.8,
                )
            axis.axhline(0, color="black", linestyle="--", linewidth=1)
            axis.set_ylim(-limit, limit)
            axis.set_title(f"{experiment} — residual at horizon t+{horizon} s")
            axis.set_ylabel("Observed − predicted\n(source units)")
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("Target time within experiment (seconds)")
        figure.savefig(plot_dir / f"{experiment}_residuals.png", dpi=150)
        figure.clear()


def save_mse_plot(metrics: pd.DataFrame, records: pd.DataFrame, run_dir: Path) -> None:
    """Plot mean squared test error by horizon, overall and per experiment."""
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    overall = metrics.sort_values("horizon_seconds")
    axis.plot(
        overall["horizon_seconds"], overall["MSE"],
        marker="o", color="black", label="All test samples", linewidth=2,
    )
    if records["experiment"].nunique() > 1:
        squared_errors = records.assign(
            squared_error=(records["observed"] - records["predicted"]).pow(2)
        )
        by_experiment = squared_errors.groupby(
            ["experiment", "horizon_seconds"], sort=True
        )["squared_error"].mean()
        for experiment in records["experiment"].unique():
            errors = by_experiment.loc[experiment]
            axis.plot(errors.index, errors, marker="o", linestyle="--", label=experiment)
    axis.set_title("Test prediction MSE by horizon")
    axis.set_xlabel("Prediction horizon (seconds)")
    axis.set_ylabel("Mean squared error (source units squared)")
    axis.set_xticks(overall["horizon_seconds"])
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(plot_dir / "mse_by_horizon.png", dpi=150)
    figure.clear()


def dataset_fingerprint(*frames: pd.DataFrame) -> str:
    """Identify the exact input tables, including schema, order, and indices."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(str((frame.shape, list(frame.columns), list(map(str, frame.dtypes)))).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def save_run(
    run_dir: Path,
    model: object,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    split_records: pd.DataFrame,
    settings: dict,
) -> None:
    """Persist model, split membership, metrics, predictions, settings, and plots."""
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    split_records.to_csv(run_dir / "split.csv", index=False)
    settings["versions"] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with (run_dir / "settings.json").open("w", encoding="utf-8") as output:
        json.dump(settings, output, indent=2, default=repr)
        output.write("\n")
    joblib.dump(model, run_dir / "model.joblib")
    save_prediction_plots(predictions, run_dir)
    save_residual_plots(predictions, run_dir)
    save_mse_plot(metrics, predictions, run_dir)
