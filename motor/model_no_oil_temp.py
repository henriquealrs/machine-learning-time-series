"""Train consumption models using all runs while excluding oil-temperature inputs."""

from pathlib import Path

import pandas as pd

from .reporting import DEFAULT_OUTPUT_DIR
from .training import OIL_TEMP_FEATURES, TrainingResult, train_model

def remove_oil_temperatures(X: pd.DataFrame) -> pd.DataFrame:
    """Drop both oil temperature and the Pi group derived from it."""
    return X.drop(columns=OIL_TEMP_FEATURES)

def train_model_ignore_oil(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    model_name: str = "hist_gradient_boosting",
    split_type: str = "experiment",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> TrainingResult:
    """Train, evaluate, plot, and record an experiment excluding oil inputs."""
    return train_model(
        X, y, metadata,
        model_name=model_name,
        split_type=split_type,
        oil_policy="ignore_oil",
        output_dir=output_dir,
    )
