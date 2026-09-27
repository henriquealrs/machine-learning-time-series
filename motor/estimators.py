"""Model-specific constructors, kept separate from training and reporting."""

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_hist_gradient_boosting() -> MultiOutputRegressor:
    """Build one boosted-tree regressor per consumption horizon.

    Trees accept NaNs and require no feature or target scaling. Disable the
    random internal validation split to keep temporal evaluation explicit.
    """
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=15,
            learning_rate=0.05,
            early_stopping=False,
            random_state=42,
        )
    )


MODEL_BUILDERS = {"hist_gradient_boosting": build_hist_gradient_boosting}


def build_model(model_name: str) -> BaseEstimator:
    """Create a fresh estimator; future models can register their own pipelines."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[model_name]()
