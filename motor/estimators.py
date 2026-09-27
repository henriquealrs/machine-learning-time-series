"""Model-specific constructors, kept separate from training and reporting."""

from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .neural_network_config import resolve_neural_network_parameters
from .targets import LogChangeRegressor


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


def build_neural_network(parameters: dict | None = None) -> TransformedTargetRegressor:
    """Build one multioutput MLP with training-only input/target preprocessing."""
    config = resolve_neural_network_parameters(parameters)
    network = MLPRegressor(
        hidden_layer_sizes=tuple(config["hidden_layers"]),
        activation=config["function"],
        solver=config["solver"],
        learning_rate_init=config["learning_rate_init"],
        alpha=config["alpha"],
        batch_size=config["batch_size"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
        early_stopping=False,
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("network", network),
    ])
    # Invert target scaling automatically so reports always use source units.
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


MODEL_BUILDERS = {
    "hist_gradient_boosting": build_hist_gradient_boosting,
    "neural_network": build_neural_network,
}


def build_model(
    model_name: str, *, target_mode: str = "absolute", log_scale: float = 1.0,
    model_parameters: dict | None = None,
) -> BaseEstimator:
    """Create a fresh estimator; future models can register their own pipelines."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {model_name!r}. Available: {list(MODEL_BUILDERS)}")
    if model_parameters is not None:
        if model_name != "neural_network":
            raise ValueError("JSON model parameters are currently supported only for neural_network.")
        estimator = build_neural_network(model_parameters)
    else:
        estimator = MODEL_BUILDERS[model_name]()
    if target_mode == "absolute":
        return estimator
    if target_mode == "log_change":
        return LogChangeRegressor(estimator, scale=log_scale)
    raise ValueError(f"Unknown target mode: {target_mode!r}")
