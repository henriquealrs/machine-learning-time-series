from pathlib import Path

import pandas as pd
import pytest

from plot_timeseries import DEFAULT_INPUT, load_slice


def test_load_slice_uses_time_column() -> None:
    frame, x_column = load_slice(
        input_path=DEFAULT_INPUT,
        sheet="Data",
        first_attribute="TachographVehicleSpeed",
        second_attribute="EngineSpeed",
        start=10,
        end=20,
    )

    assert x_column == "Time[s]"
    assert len(frame) == 10
    assert list(frame.columns) == [
        "Time[s]",
        "TachographVehicleSpeed",
        "EngineSpeed",
    ]


def test_load_slice_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="Missing column"):
        load_slice(
            input_path=DEFAULT_INPUT,
            sheet="Data",
            first_attribute="not-a-column",
            second_attribute="EngineSpeed",
            start=0,
            end=10,
        )


def test_load_slice_adds_sample_axis_without_time(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.xlsx"
    pd.DataFrame({"first": [1, 2, 3], "second": [4, 5, 6]}).to_excel(
        input_path, index=False
    )

    frame, x_column = load_slice(
        input_path=input_path,
        sheet="Sheet1",
        first_attribute="first",
        second_attribute="second",
        start=1,
        end=3,
    )

    assert x_column == "Sample"
    assert frame["Sample"].tolist() == [1, 2]

