"""Build Buckingham Pi groups from vehicle measurements and plot regressions."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scist
from numpy.typing import NDArray


MILEAGE = "1000GD"
EXCEL_FILE = Path("data/1000GD.xlsx")
PARAMETERS_SHEET = 1
Y_AXIS_INDEX = 6
INITIAL_ROW = 0
FINAL_ROW = 3852


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata and dimensional exponents for one physical parameter."""

    alias: str
    name: str
    si_unit: str
    factor: float
    dimensions: dict[str, float]
    repeatable: bool


@dataclass(frozen=True)
class PiGroup:
    """A named dimensionless group and the powers of its variables."""

    name: str
    powers: tuple[tuple[str, float], ...]


def load_workbook(
    excel_file: Path, parameters_sheet: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the measurement and parameter sheets from an Excel workbook."""
    sheets = pd.read_excel(excel_file, sheet_name=[0, parameters_sheet])
    return sheets[0], sheets[parameters_sheet]


def find_fundamental_dimensions(parameters: pd.DataFrame) -> list[str]:
    """Return one-letter columns such as M, L, T, and K in sheet order."""
    return [str(column) for column in parameters.columns if len(str(column)) == 1]


def parse_parameter_specs(
    parameters: pd.DataFrame, fundamental_dimensions: list[str]
) -> list[ParameterSpec]:
    """Convert each parameter-table row into a typed metadata object."""
    specs: list[ParameterSpec] = []

    for _, row in parameters.iterrows():
        specs.append(
            ParameterSpec(
                alias=str(row["Alias"]),
                name=str(row["Parameter"]),
                si_unit=str(row["SI"]),
                factor=float(row["Factor"]),
                dimensions={
                    dimension: float(row[dimension])
                    for dimension in fundamental_dimensions
                },
                repeatable=row["Repeatable"] == "ok",
            )
        )

    return specs


def print_inputs(data: pd.DataFrame, parameters: pd.DataFrame) -> None:
    """Display the workbook contents, matching the original script's output."""
    print("\nAll the data:")
    print(data)
    print("\n\nParameters Table:")
    print(parameters)


def print_parameter_specs(
    specs: list[ParameterSpec], fundamental_dimensions: list[str]
) -> None:
    """Display the metadata collected for every parameter."""
    print(
        "\nFundamental Dimensions:\n",
        fundamental_dimensions,
        "\n",
    )

    for spec in specs:
        info = [
            {dimension: spec.dimensions[dimension]}
            for dimension in fundamental_dimensions
        ]
        info.extend(
            [
                {"Factor": spec.factor},
                {"Parameter": spec.name},
                {"Alias": spec.alias},
            ]
        )
        print(spec.name, "- variable name:", spec.alias, "\n", info, "\n")


def select_repeatable_parameters(specs: list[ParameterSpec]) -> list[ParameterSpec]:
    """Select parameters marked ``ok`` in the Repeatable column."""
    repeatable = [spec for spec in specs if spec.repeatable]

    print("The repeatable variables are:")
    for spec in repeatable:
        print(spec.alias, "-", spec.name)

    return repeatable


def build_dimensional_matrix(
    repeatable: list[ParameterSpec], fundamental_dimensions: list[str]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Build A and remove dimensions absent from every repeatable parameter.

    The returned reduced matrix has dimensions as rows and repeatable
    parameters as columns. The full matrix retains every fundamental dimension
    and is used to detect incompatible non-repeatable parameters.
    """
    full_matrix = np.asarray(
        [
            [spec.dimensions[dimension] for dimension in fundamental_dimensions]
            for spec in repeatable
        ],
        dtype=np.float64,
    )
    active_dimensions: list[str] = []
    active_columns: list[int] = []

    for column_index, dimension in enumerate(fundamental_dimensions):
        column_is_zero = np.all(full_matrix[:, column_index] == 0)
        if column_is_zero:
            print(
                "The column relative to",
                dimension,
                "dimension was deleted from A because it is null.",
            )
        else:
            print("The column relative to", dimension, "has values")
            active_dimensions.append(dimension)
            active_columns.append(column_index)

    matrix = full_matrix[:, active_columns].T
    print(
        "\nA matrix legend\nRow order:",
        active_dimensions,
        "\nColumn order:",
        [spec.alias for spec in repeatable],
    )
    print("\n", matrix)

    return matrix, full_matrix, active_dimensions


def target_is_compatible(
    target: ParameterSpec,
    full_matrix: NDArray[np.float64],
    fundamental_dimensions: list[str],
) -> bool:
    """Check that repeatable parameters span every dimension used by target."""
    compatible = True

    for column_index, dimension in enumerate(fundamental_dimensions):
        repeatable_dimension_is_zero = np.all(full_matrix[:, column_index] == 0)
        if repeatable_dimension_is_zero and target.dimensions[dimension] != 0:
            print(
                "It is not possible to create a Pi with parameter",
                target.alias,
                "because the repeatable parameters do not have",
                dimension,
                "dimension, unlike this parameter.",
            )
            compatible = False

    return compatible


def solve_pi_groups(
    specs: list[ParameterSpec],
    repeatable: list[ParameterSpec],
    matrix: NDArray[np.float64],
    full_matrix: NDArray[np.float64],
    active_dimensions: list[str],
    fundamental_dimensions: list[str],
) -> list[PiGroup]:
    """Solve one Buckingham Pi group for each non-repeatable parameter."""
    repeatable_aliases = {spec.alias for spec in repeatable}
    targets = [spec for spec in specs if spec.alias not in repeatable_aliases]
    pi_groups: list[PiGroup] = []

    for pi_number, target in enumerate(targets, start=1):
        print(target.alias)

        if not target_is_compatible(
            target, full_matrix, fundamental_dimensions
        ):
            continue

        target_dimensions = np.asarray(
            [-target.dimensions[dimension] for dimension in active_dimensions],
            dtype=np.float64,
        )
        exponents = np.linalg.solve(matrix, target_dimensions)

        if np.all(exponents == 0):
            print(
                "It is not possible to create a Pi with parameter",
                target.alias,
                "because the only solution to the matrix results in null powers.",
            )
            continue

        print("\nPi", pi_number, ": ")
        powers: list[tuple[str, float]] = []
        for repeatable_spec, exponent in zip(repeatable, exponents, strict=True):
            print(repeatable_spec.alias, "^", exponent)
            powers.append((repeatable_spec.alias, exponent))

        print(target.alias, "^ 1", "\n")
        powers.append((target.alias, 1))
        pi_groups.append(PiGroup(name=f"Pi{pi_number}", powers=tuple(powers)))

    print(
        [
            {group.name: [{alias: power} for alias, power in group.powers]}
            for group in pi_groups
        ]
    )
    return pi_groups


def replace_aliases_with_column_names(
    pi_groups: list[PiGroup], specs: list[ParameterSpec]
) -> list[PiGroup]:
    """Replace short aliases in each Pi group with measurement column names."""
    column_name_by_alias = {spec.alias: spec.name for spec in specs}
    return [
        PiGroup(
            name=group.name,
            powers=tuple(
                (column_name_by_alias[alias], power)
                for alias, power in group.powers
            ),
        )
        for group in pi_groups
    ]


def calculate_pi_column(data: pd.DataFrame, pi_group: PiGroup) -> None:
    """Calculate one dimensionless Pi column in place."""
    for index, (column_name, power) in enumerate(pi_group.powers):
        powered_values = data[column_name] ** power
        if index == 0:
            data[pi_group.name] = powered_values
        else:
            data[pi_group.name] *= powered_values


def plot_pi_regression(
    data: pd.DataFrame,
    pi_group: PiGroup,
    y_axis: str,
    mileage: str,
    initial_row: int,
    final_row: int,
) -> None:
    """Plot one Pi group against the selected response and its fitted line."""
    selected_rows = data.loc[initial_row:final_row]
    x = selected_rows[pi_group.name]
    y = selected_rows[y_axis]
    regression = scist.linregress(x, y)

    plt.plot(x, y, "o", label=f"Original data - {mileage}")
    plt.plot(
        x,
        regression.intercept + regression.slope * x,
        "r",
        label="Fitted line",
    )
    plt.xlabel(
        f"{pi_group.name}  rows({initial_row}-{final_row})",
        loc="center",
    )
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid()
    print("With trend line.\nR²: ", regression.rvalue**2)
    plt.show()


def main() -> None:
    """Run the complete dimensional-analysis and plotting workflow."""
    data, parameter_table = load_workbook(EXCEL_FILE, PARAMETERS_SHEET)
    print_inputs(data, parameter_table)

    fundamental_dimensions = find_fundamental_dimensions(parameter_table)
    specs = parse_parameter_specs(parameter_table, fundamental_dimensions)
    print_parameter_specs(specs, fundamental_dimensions)

    repeatable = select_repeatable_parameters(specs)
    matrix, full_matrix, active_dimensions = build_dimensional_matrix(
        repeatable, fundamental_dimensions
    )
    pi_groups = solve_pi_groups(
        specs,
        repeatable,
        matrix,
        full_matrix,
        active_dimensions,
        fundamental_dimensions,
    )
    named_pi_groups = replace_aliases_with_column_names(pi_groups, specs)

    y_axis = str(data.columns[Y_AXIS_INDEX])
    for pi_group in named_pi_groups:
        calculate_pi_column(data, pi_group)
        plot_pi_regression(
            data,
            pi_group,
            y_axis,
            MILEAGE,
            INITIAL_ROW,
            FINAL_ROW,
        )


if __name__ == "__main__":
    main()
