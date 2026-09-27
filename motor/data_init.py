"""Feature scaling for the combined experiment table."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit one scaler on the supplied X table and return its scaled copy.

    Preserve row indices, column names, and missing values.
    Return the fitted scaler for transforming additional data consistently.
    All supplied samples contribute to fitting: supply only training data
    when evaluating on held-out experiments or time intervals.
    """
    scaler = StandardScaler().set_output(transform="pandas")
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler


def _split_by_experiment(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out the smallest present experiment; ties use first appearance."""
    sizes = metadata.groupby("experiment", sort=False).size()
    if len(sizes) < 2:
        raise ValueError("Experiment splitting requires at least two nonempty experiments.")
    test_mask = metadata["experiment"].eq(sizes.idxmin())
    train_mask = ~test_mask
    return X.loc[train_mask, :], y.loc[train_mask, :], X.loc[test_mask, :], y.loc[test_mask, :]


def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metadata: pd.DataFrame,
    split_type: str = "experiment",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch the selected split strategy without silently returning empty data."""
    if not (X.index.equals(y.index) and X.index.equals(metadata.index)):
        raise ValueError("X, y, and metadata must have matching row indices.")
    if split_type == "experiment":
        return _split_by_experiment(X, y, metadata)
    # Add future split strategies here; reporting already separates their runs.
    raise ValueError(f"Unknown split type: {split_type!r}")
