"""Load and validate basic neural-network parameters from JSON."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "neural_network.json"


@dataclass
class NeuralNetworkConfig:
    """User-facing parameter names for a small scikit-learn MLP."""

    hidden_layers: list[int]
    function: str
    solver: str
    learning_rate_init: float
    alpha: float
    batch_size: str | int
    max_iter: int
    tol: float
    random_state: int
    clamp_output: bool = False

    def __post_init__(self) -> None:
        if type(self.clamp_output) is not bool:
            raise ValueError("clamp_output must be a boolean.")
        if not isinstance(self.hidden_layers, list) or not self.hidden_layers or any(
            type(size) is not int or size <= 0 for size in self.hidden_layers
        ):
            raise ValueError("hidden_layers must be a nonempty list of positive integers.")
        if self.function not in ("identity", "logistic", "tanh", "relu"):
            raise ValueError("function must be identity, logistic, tanh, or relu.")
        if self.solver not in ("adam", "sgd", "lbfgs"):
            raise ValueError("solver must be adam, sgd, or lbfgs.")
        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.batch_size != "auto" and (
            type(self.batch_size) is not int or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be 'auto' or a positive integer.")
        if type(self.random_state) is not int or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")
        for name in ("learning_rate_init", "alpha", "tol"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            minimum_valid = value >= 0 if name == "alpha" else value > 0
            if not minimum_valid:
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'alpha' else 'positive'}.")


def _read_config(path: Path) -> dict:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Neural-network configuration must be a JSON object.")
    unknown = set(content) - {field.name for field in fields(NeuralNetworkConfig)}
    if unknown:
        raise ValueError(f"Unknown neural-network parameters: {sorted(unknown)}")
    return content


def load_neural_network_config(path: str | Path = DEFAULT_CONFIG_PATH) -> NeuralNetworkConfig:
    """Apply an optional alternative file's overrides to the bundled defaults."""
    defaults = _read_config(DEFAULT_CONFIG_PATH)
    path = Path(path)
    if path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        defaults.update(_read_config(path))
    return NeuralNetworkConfig(**defaults)


def resolve_neural_network_parameters(parameters: dict | None = None) -> dict:
    """Merge overrides and return validated effective parameters for recording."""
    defaults = asdict(load_neural_network_config())
    defaults.update(parameters or {})
    return asdict(NeuralNetworkConfig(**defaults))
