"""Entry point for the fuel-consumption model workflow."""

import argparse
from pathlib import Path

from .data_init import scale_features
from .data_io import DEFAULT_DATA_DIR
from .dataset import load_dataset


def main() -> None:
    """Load the dataset for the model's initial preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-horizon", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.max_horizon < 0:
        parser.error("--max-horizon must be nonnegative")

    features, targets, metadata = load_dataset(
        data_dir=arguments.data_dir,
        max_horizon=arguments.max_horizon,
    )
    print(features.columns)

    print("Combined model dataset loaded.")
    print(f"X: {features.shape[0]} samples x {features.shape[1]} features")
    print(f"y: {targets.shape[0]} samples x {targets.shape[1]} targets")
    print("Samples per experiment:")
    print(metadata["experiment"].value_counts().to_string())
    print(f"Prediction horizons: 0..{arguments.max_horizon} seconds")

    scaled_features, _ = scale_features(features)
    print(scaled_features.head())

if __name__ == "__main__":
    main()
