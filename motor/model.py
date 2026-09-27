"""Entry point for the fuel-consumption model workflow."""

import argparse
from pathlib import Path

from .data_io import DEFAULT_DATA_DIR
from .dataset import load_dataset
from .estimators import MODEL_BUILDERS
from .model_no_oil_temp import train_model_ignore_oil
from .reporting import DEFAULT_OUTPUT_DIR
from .training import train_model


def main() -> None:
    """Load data, train the selected model, and save a separate experiment run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), default="hist_gradient_boosting")
    parser.add_argument("--split-type", default="experiment")
    parser.add_argument("--oil-policy", choices=["ignore_oil", "with_oil"], default="ignore_oil")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    if arguments.max_horizon < 0:
        parser.error("--max-horizon must be nonnegative")

    features, targets, metadata = load_dataset(
        data_dir=arguments.data_dir,
        max_horizon=arguments.max_horizon,
    )
    print("Combined model dataset loaded.")
    print(f"X: {features.shape[0]} samples x {features.shape[1]} features")
    print(f"y: {targets.shape[0]} samples x {targets.shape[1]} targets")
    print("Samples per experiment:")
    print(metadata["experiment"].value_counts().to_string())
    print(f"Prediction horizons: 0..{arguments.max_horizon} seconds")

    if arguments.oil_policy == "ignore_oil":
        result = train_model_ignore_oil(
            features, targets, metadata,
            model_name=arguments.model,
            split_type=arguments.split_type,
            output_dir=arguments.output_dir,
        )
    else:
        result = train_model(
            features, targets, metadata,
            model_name=arguments.model,
            split_type=arguments.split_type,
            oil_policy="with_oil",
            output_dir=arguments.output_dir,
        )
    print(result.metrics.to_string(index=False))
    print(f"Saved model run: {result.output_dir}")


if __name__ == "__main__":
    main()
