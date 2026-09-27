"""Entry point for the fuel-consumption model workflow."""

import argparse
from pathlib import Path

from motor.data_init import scale_features

from .data_io import DEFAULT_DATA_DIR, EXPERIMENTS
from .dataset import load_dataset, ExperimentDatasets


def main() -> None:
    """Load the dataset for the model's initial preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-horizon", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.max_horizon < 0:
        parser.error("--max-horizon must be nonnegative")

    full_data = load_dataset(
        data_dir=arguments.data_dir,
        max_horizon=arguments.max_horizon,
    )
    print(full_data.d1t1a[0].columns)

    print("Independent model datasets loaded.")
    for experiment, (features, targets) in zip(
        EXPERIMENTS, full_data, strict=True
    ):
        print(
            f"{experiment}: X={features.shape}, y={targets.shape} "
            f"({features.isna().sum().sum()} missing feature values)"
        )
    print(f"Prediction horizons: 0..{arguments.max_horizon} seconds")

    scaled_data, _ = scale_features(full_data)
    for data in scaled_data:
        print(data[0].head())




if __name__ == "__main__":
    main()
