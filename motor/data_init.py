"""Feature scaling for the combined experiment table."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from .data_io import EXPERIMENTS

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
		metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame,  pd.DataFrame]:
    masks = [metadata["experiment"] == exp for exp in EXPERIMENTS]
    sizes = np.array([mask.sum() for mask in masks])
    idx_min = np.argmin(sizes)

    train_mask = ~masks[idx_min]
    Xtrain = X.loc[train_mask, :]
    ytrain = y.loc[train_mask, :]

    test_mask = ~train_mask
    Xtest = X.loc[test_mask, :]
    ytest = y.loc[test_mask, :]

    return (Xtrain, ytrain, Xtest, ytest)

def split_data(X: pd.DataFrame, y: pd.DataFrame, metadata: pd.DataFrame, split_type: str = "experiment") -> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame,  pd.DataFrame]:
    if split_type == "experiment":
        return _split_by_experiment(X, y, metadata)
    # TODO other path
    return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
