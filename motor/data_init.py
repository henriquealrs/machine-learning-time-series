"""Feature scaling shared across the independent experiment datasets."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import ExperimentDatasets


def scale_features(
    datasets: ExperimentDatasets,
) -> tuple[ExperimentDatasets, StandardScaler]:
    """Fit one scaler on all X tables and transform each experiment separately.

    Preserve sample indices, column names, missing values, and unscaled y.
    Return the fitted scaler for transforming additional data consistently.
    All supplied samples contribute to fitting: supply only training data
    when evaluating on held-out experiments or time intervals.
    """
    columns = datasets[0][0].columns
    if any(not features.columns.equals(columns) for features, _ in datasets):
        raise ValueError("All experiments must have the same feature columns in order.")

    combined_features = pd.concat(
        [features for features, _ in datasets], ignore_index=True
    )
    scaler = StandardScaler().set_output(transform="pandas")
    scaler.fit(combined_features)

    scaled_datasets = ExperimentDatasets(*(
        (scaler.transform(features), targets)
        for features, targets in datasets
    ))
    return scaled_datasets, scaler
