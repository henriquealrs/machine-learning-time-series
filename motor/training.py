"""Shared training workflow for different models, splits, and oil policies."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from .data_init import split_data
from .estimators import build_model
from .reporting import (
    DEFAULT_OUTPUT_DIR,
    create_run_directory,
    dataset_fingerprint,
    evaluate_predictions,
    prediction_records,
    save_run,
)


OIL_TEMP_FEATURES = ["oil_temperature", "pi_7"]
OilPolicy = Literal["ignore_oil", "with_oil"]


@dataclass
class TrainingResult:
    """Trained estimator, test predictions, metrics, and saved run directory."""

    model: BaseEstimator
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    output_dir: Path


def select_oil_policy(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    oil_policy: OilPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ignore oil features or retain only rows with available oil features."""
    if oil_policy == "ignore_oil":
        return features.drop(columns=OIL_TEMP_FEATURES), targets, metadata
    if oil_policy == "with_oil":
        valid = np.isfinite(features[OIL_TEMP_FEATURES]).all(axis=1)
        return features.loc[valid], targets.loc[valid], metadata.loc[valid]
    raise ValueError(f"Unknown oil policy: {oil_policy!r}")


def train_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    model_name: str = "hist_gradient_boosting",
    split_type: str = "experiment",
    oil_policy: OilPolicy = "ignore_oil",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_mode: Literal["absolute", "log_change"] = "absolute",
    log_scale: float = 1.0,
    model_parameters: dict | None = None,
) -> TrainingResult:
    """Train and save a new run; model-specific preprocessing belongs in its builder."""
    if not (
        features.index.is_unique
        and features.index.equals(targets.index)
        and features.index.equals(metadata.index)
    ):
        raise ValueError("X, y, and metadata must have identical unique row indices.")
    if target_mode == "log_change" and oil_policy != "with_oil":
        raise ValueError("Log-change experiments currently require oil_policy='with_oil'.")
    model = build_model(
        model_name, target_mode=target_mode, log_scale=log_scale,
        model_parameters=model_parameters,
    )
    input_fingerprint = dataset_fingerprint(features, targets, metadata)
    original_count = len(features)
    original_metadata = metadata.copy()
    features, targets, metadata = select_oil_policy(features, targets, metadata, oil_policy)
    X_train, y_train, X_test, y_test = split_data(features, targets, metadata, split_type)

    started = perf_counter()
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    training_seconds = perf_counter() - started
    predictions = pd.DataFrame(
        model.predict(X_test), index=y_test.index, columns=y_test.columns
    )
    metrics = evaluate_predictions(y_test, predictions)
    records = prediction_records(y_test, predictions, metadata)

    split_records = original_metadata
    split_records.insert(0, "row_id", split_records.index)
    split_records["partition"] = "excluded"
    split_records.loc[X_train.index, "partition"] = "train"
    split_records.loc[X_test.index, "partition"] = "test"

    settings = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_class": type(model).__name__,
        "parameters": model.get_params(deep=True),
        "model_configuration": model_parameters,
        "fit_warnings": [str(warning.message) for warning in fit_warnings],
        "split_type": split_type,
        "split_rule": "Smallest nonempty experiment is held out" if split_type == "experiment" else split_type,
        "oil_policy": oil_policy,
        "target_mode": target_mode,
        "log_scale": log_scale if target_mode == "log_change" else None,
        "features": features.columns.tolist(),
        "targets": targets.columns.tolist(),
        "input_samples": original_count,
        "oil_policy_excluded_samples": original_count - len(features),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_experiments": metadata.loc[X_train.index, "experiment"].unique().tolist(),
        "test_experiments": metadata.loc[X_test.index, "experiment"].unique().tolist(),
        "training_seconds": training_seconds,
        "input_sha256": input_fingerprint,
        "scaling": "Defined by the model builder; no external scaling applied",
        "target_units": "Original source units; physical unit requires confirmation",
    }
    run_dir = create_run_directory(Path(output_dir), model_name, split_type, oil_policy, target_mode)
    save_run(run_dir, model, metrics, records, split_records, settings)
    return TrainingResult(model, predictions, metrics, run_dir)
