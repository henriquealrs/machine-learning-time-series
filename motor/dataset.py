"""Build aligned feature, target, and metadata tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import (
    DEFAULT_DATA_DIR,
    SAMPLE_KEYS,
    load_measurements,
    load_pi_groups,
)


# Omit injection rate and mass because they are absent in entire experiments.
# Fuel consumption must never be a feature when estimating the current value.
RAW_FEATURES = [
    "vehicle_speed", "accelerator_position", "brake_position", "gear",
    "engine_speed", "engine_torque", "coolant_temperature", "oil_temperature",
    "ambient_temperature",
]
# pi_1..pi_3 duplicate the corresponding controls; retained for comparison.
# pi_4 includes elapsed time. pi_6 contains the target and is excluded.
PI_FEATURES = ["pi_1", "pi_2", "pi_3", "pi_4", "pi_7", "pi_8"]

def target_columns(max_horizon: int) -> list[str]:
    """Name pointwise consumption targets from t0 through the final horizon."""
    return [f"fuel_consumption_t_plus_{horizon}" for horizon in range(max_horizon + 1)]


def build_targets(measurements: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    """Shift consumption within each experiment, validating elapsed seconds."""
    if max_horizon < 0:
        raise ValueError("max_horizon must be nonnegative.")
    frames = []
    for _, experiment in measurements.groupby("experiment", sort=False):
        frame = experiment.sort_values("sample").copy()
        for horizon, column in enumerate(target_columns(max_horizon)):
            future_time = frame["time_seconds"].shift(-horizon)
            valid_interval = np.isclose(
                future_time - frame["time_seconds"], horizon
            )
            frame[column] = frame["fuel_consumption"].shift(-horizon).where(
                valid_interval
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def align_pi_groups(samples: pd.DataFrame, pi_groups: pd.DataFrame) -> pd.DataFrame:
    """Join Pi groups by sample identity and reject missing or stale matches."""
    aligned = samples.merge(
        pi_groups[SAMPLE_KEYS + ["time_seconds"] + PI_FEATURES],
        on=SAMPLE_KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_pi"),
        indicator=True,
    )
    matching_times = np.isclose(
        aligned["time_seconds"], aligned["time_seconds_pi"], equal_nan=True
    )
    if not (aligned["_merge"].eq("both").all() and matching_times.all()):
        raise ValueError("Pi samples do not match measurements. Regenerate the Pi dataset.")
    return aligned.drop(columns=["_merge", "time_seconds_pi"])


def load_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    max_horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return combined X, y, and metadata with matching row indices.

    X contains current measurements and selected Pi groups; missing features
    remain NaN. y contains pointwise consumption at t0..t+max_horizon. Rows
    without complete finite targets are discarded within each experiment.
    Metadata identifies each row's experiment, original sample, and time.
    Targets are built separately within each experiment before combining.
    No lag features, scaling, or imputation are performed.
    """
    data_dir = Path(data_dir)
    if max_horizon < 0:
        raise ValueError("max_horizon must be nonnegative.")
    samples = build_targets(load_measurements(data_dir), max_horizon)
    samples = align_pi_groups(samples, load_pi_groups(data_dir))
    targets = target_columns(max_horizon)
    samples = samples.replace([np.inf, -np.inf], np.nan)
    samples = samples.dropna(subset=targets).reset_index(drop=True)
    if samples.empty:
        raise ValueError("No complete consumption windows were found.")
    features = samples[RAW_FEATURES + PI_FEATURES].copy()
    responses = samples[targets].copy()
    metadata = samples[SAMPLE_KEYS + ["time_seconds"]].copy()
    return features, responses, metadata
