"""Target transformations with consumption-scale prediction reconstruction."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


CURRENT_CONSUMPTION = "current_fuel_consumption"


class LogChangeRegressor(RegressorMixin, BaseEstimator):
    """Fit log1p(future/scale) - log1p(current/scale), predict consumption.

    Current measured consumption must be available in X at prediction time.
    The positive scale has the same units as consumption. Reconstructed
    negative consumption is clipped to zero; no future measurements enter X.
    """

    def __init__(self, estimator: BaseEstimator, scale: float = 1.0):
        self.estimator = estimator
        self.scale = scale

    def _reference(self, X: pd.DataFrame) -> np.ndarray:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Log-change scale must be finite and positive.")
        reference = X[CURRENT_CONSUMPTION].to_numpy(dtype=float)
        if not np.isfinite(reference).all() or (reference < 0).any():
            raise ValueError("Current consumption must be finite and nonnegative.")
        return np.log1p(reference / self.scale)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """Fit the underlying regressor on log changes using training rows only."""
        reference = self._reference(X)
        if not np.isfinite(y.to_numpy()).all() or (y.to_numpy() < 0).any():
            raise ValueError("Future consumption must be finite and nonnegative.")
        log_changes = np.log1p(y / self.scale).sub(reference, axis=0)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, log_changes)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_log_change(self, X: pd.DataFrame) -> np.ndarray:
        """Return the raw log-change predictions before reconstruction/clipping."""
        check_is_fitted(self, "estimator_")
        predictions = np.asarray(self.estimator_.predict(X))
        return predictions[:, None] if predictions.ndim == 1 else predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Reconstruct nonnegative consumption for the supplied future horizons."""
        reference = self._reference(X)
        changes = self.predict_log_change(X)
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = self.scale * np.expm1(reference[:, None] + changes)
        if not np.isfinite(predictions).all():
            raise ValueError("Log-change predictions produced nonfinite consumption.")
        return np.maximum(predictions, 0.0)
