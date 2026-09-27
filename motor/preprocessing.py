"""Run the initial Pi analysis and transformation pipeline."""

import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]


def prepare_data(project_dir: Path = PROJECT_DIR) -> None:
    """Generate Pi datasets, transformed columns, and plots from raw CSVs.

    Run the existing pipeline in a separate process with the repository as
    its working directory, as required by the original scripts.
    """
    project_dir = Path(project_dir).resolve()
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, str(project_dir / "Main_Motor.py")],
        cwd=project_dir,
        env=environment,
        check=True,
    )


def main() -> None:
    """Execute the initial preparation command."""
    prepare_data()


if __name__ == "__main__":
    main()
