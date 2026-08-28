"""Plot a slice of two attributes from an Excel time-series worksheet."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "1oDia_Teste2.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "speed_and_engine_speed.png"
DEFAULT_SHEET = "Data"
DEFAULT_FIRST_ATTRIBUTE = "TachographVehicleSpeed"
DEFAULT_SECOND_ATTRIBUTE = "EngineSpeed"
TIME_COLUMN = "Time[s]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a row slice of two attributes from an Excel worksheet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--first", default=DEFAULT_FIRST_ATTRIBUTE)
    parser.add_argument("--second", default=DEFAULT_SECOND_ATTRIBUTE)
    parser.add_argument("--start", type=int, default=0, help="First row to plot.")
    parser.add_argument(
        "--end", type=int, default=500, help="Exclusive final row to plot."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="Save the chart without opening an interactive window.",
    )
    return parser.parse_args()


def load_slice(
    input_path: Path,
    sheet: str,
    first_attribute: str,
    second_attribute: str,
    start: int,
    end: int,
) -> tuple[pd.DataFrame, str]:
    """Load the requested columns and return the selected row range."""
    if start < 0 or end <= start:
        raise ValueError("The row range must satisfy 0 <= start < end.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {input_path}")

    requested = [first_attribute, second_attribute]
    header = pd.read_excel(input_path, sheet_name=sheet, nrows=0)
    missing = [column for column in requested if column not in header.columns]
    if missing:
        available = ", ".join(map(str, header.columns))
        raise ValueError(
            f"Missing column(s): {', '.join(missing)}. Available columns: {available}"
        )

    use_columns = requested.copy()
    x_column = "Sample"
    if TIME_COLUMN in header.columns:
        use_columns.insert(0, TIME_COLUMN)
        x_column = TIME_COLUMN

    frame = pd.read_excel(input_path, sheet_name=sheet, usecols=use_columns)
    frame = frame.iloc[start:end].copy()
    if frame.empty:
        raise ValueError(f"The selected row range {start}:{end} contains no data.")

    if x_column == "Sample":
        frame.insert(0, x_column, frame.index)

    return frame, x_column


def create_plot(
    frame: pd.DataFrame,
    x_column: str,
    first_attribute: str,
    second_attribute: str,
    output_path: Path,
    show: bool,
) -> None:
    """Create a dual-axis line chart and save it as a PNG file."""
    if show:
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise RuntimeError(
                "No graphical display was detected. Run again with --no-show "
                "to save the chart without opening a window."
            )

        try:
            import PyQt6  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "The PyQt6 graphical backend is unavailable. Run 'uv sync' "
                "and try again, or use --no-show."
            ) from error

        matplotlib.use("QtAgg", force=True)
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure, first_axis = plt.subplots(figsize=(12, 6))
    second_axis = first_axis.twinx()

    first_line = first_axis.plot(
        frame[x_column],
        frame[first_attribute],
        color="tab:blue",
        linewidth=1.5,
        label=first_attribute,
    )
    second_line = second_axis.plot(
        frame[x_column],
        frame[second_attribute],
        color="tab:orange",
        linewidth=1.3,
        alpha=0.85,
        label=second_attribute,
    )

    first_axis.set_title(f"{first_attribute} and {second_attribute}")
    first_axis.set_xlabel(x_column)
    first_axis.set_ylabel(first_attribute, color="tab:blue")
    second_axis.set_ylabel(second_attribute, color="tab:orange")
    first_axis.tick_params(axis="y", labelcolor="tab:blue")
    second_axis.tick_params(axis="y", labelcolor="tab:orange")
    first_axis.grid(alpha=0.25)
    first_axis.legend(
        first_line + second_line,
        [line.get_label() for line in first_line + second_line],
        loc="upper right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frame, x_column = load_slice(
        input_path=args.input,
        sheet=args.sheet,
        first_attribute=args.first,
        second_attribute=args.second,
        start=args.start,
        end=args.end,
    )
    create_plot(
        frame=frame,
        x_column=x_column,
        first_attribute=args.first,
        second_attribute=args.second,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
