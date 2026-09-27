"""Command-line training for a configurable basic multioutput neural network."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data_io import DEFAULT_DATA_DIR
from .dataset import load_dataset
from .neural_network_config import DEFAULT_CONFIG_PATH, load_neural_network_config
from .reporting import DEFAULT_OUTPUT_DIR
from .training import train_model


def main() -> None:
    """Read JSON parameters and save a neural-network experiment run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--split-type", default="experiment")
    parser.add_argument("--oil-policy", choices=["ignore_oil", "with_oil"], default="with_oil")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    if arguments.max_horizon < 0:
        parser.error("--max-horizon must be nonnegative")
    try:
        config = load_neural_network_config(arguments.config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    parameters = asdict(config)
    features, targets, metadata = load_dataset(arguments.data_dir, arguments.max_horizon)
    result = train_model(
        features, targets, metadata,
        model_name="neural_network",
        model_parameters=parameters,
        split_type=arguments.split_type,
        oil_policy=arguments.oil_policy,
        output_dir=arguments.output_dir,
    )
    (result.output_dir / "config.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    settings_path = result.output_dir / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    network = result.model.regressor_.named_steps["network"]
    settings.update({
        "configuration_file": str(arguments.config.resolve()),
        "scaling": "Training-only median imputation and StandardScaler on X; StandardScaler on y with automatic inverse transform",
        "training_iterations": int(network.n_iter_),
        "final_training_loss": float(network.loss_),
    })
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Network hidden layers: {config.hidden_layers}; activation: {config.function}")
    print(result.metrics.to_string(index=False))
    for warning in settings["fit_warnings"]:
        print(f"Training warning: {warning}")
    print(f"Saved neural-network run: {result.output_dir}")


if __name__ == "__main__":
    main()
