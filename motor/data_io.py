"""Read legacy CSV files and translate their schema at the input boundary."""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "Dados"
EXPERIMENTS = ("D1T1A", "D1T1B", "D1T2", "D2T1")
VARIABLE_NAMES = {
    "Temp": "time_seconds",
    "Velo": "vehicle_speed",
    "Acc": "accelerator_position",
    "Brake": "brake_position",
    "Gear": "gear",
    "n": "engine_speed",
    "Tau": "engine_torque",
    "Diem": "injection_rate",
    "Taccm": "fuel_consumption",
    "TeLam": "coolant_temperature",
    "Teom": "oil_temperature",
    "Team": "ambient_temperature",
    "x": "vehicle_distance",
    "zAl": "altitude",
    "yLa": "latitude_distance",
    "xLo": "longitude_distance",
    "mass": "axle_mass",
}
SAMPLE_KEYS = ["experiment", "sample"]


def read_csv(path: Path) -> pd.DataFrame:
    """Read the semicolon-separated, decimal-comma source format."""
    frame = pd.read_csv(path, sep=";", decimal=",", encoding="latin-1")
    frame.columns = frame.columns.str.strip()
    return frame


def load_measurements(data_dir: Path) -> pd.DataFrame:
    """Convert wide experiment columns into one row per observed sample."""
    measurements = read_csv(data_dir / "Data_set.CSV")
    parameters = read_csv(data_dir / "Data_set_param.CSV")
    column_map = dict(zip(
        parameters["Alias"].str.strip(),
        parameters["Colunas - Parametros"].str.strip(),
        strict=True,
    ))
    experiments = []
    for experiment in EXPERIMENTS:
        # Preserve the Pi generator's sample numbering: remove only rows in
        # which all parameter variables are missing, before selecting features.
        frame = pd.DataFrame({
            alias: pd.to_numeric(
                measurements[f"{experiment} - {column}"], errors="coerce"
            )
            for alias, column in column_map.items()
        }).dropna(how="all").reset_index(drop=True)
        frame = frame.rename(columns=VARIABLE_NAMES)
        frame.insert(0, "sample", np.arange(len(frame)))
        frame.insert(0, "experiment", experiment)
        experiments.append(frame)
    return pd.concat(experiments, ignore_index=True)


def load_pi_groups(data_dir: Path) -> pd.DataFrame:
    """Read previously generated Pi groups with English metadata names."""
    path = data_dir / "Dataset_Dimensional_Final.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Pi dataset: {path}. Run 'uv run prepare-data' first."
        )
    frame = read_csv(path)
    return frame.rename(columns={
        "Experimento": "experiment",
        "Amostra": "sample",
        "Temp": "time_seconds",
        **{column: column.lower() for column in frame if column.startswith("Pi_")},
    })
