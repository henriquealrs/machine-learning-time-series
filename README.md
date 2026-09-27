# Motor

Python scripts for Buckingham Pi analysis and plotting of motor experiment data.

## Setup and run

From the repository root, install the locked dependencies with uv:

```bash
uv sync --locked
uv run prepare-data
```

The environment lives in `.venv/`. Python 3.14 is selected by
`.python-version`; uv downloads it if needed.

Run from the repository root because the scripts locate `Auxi/` and `Dados/`
relative to the current working directory. The pipeline reads `Dados/Data_set.CSV`
and `Dados/Data_set_param.CSV`, then writes generated CSVs and PNG plots under
`Dados/`. Running it again overwrites those generated outputs.

For a headless session:

```bash
MPLBACKEND=Agg uv run prepare-data
```

To activate the environment manually:

```bash
source .venv/bin/activate
```

## Prepare the supervised dataset

Generate the Pi files with `uv run prepare-data` first, then run:

```bash
uv run python -m motor
```

The `motor/` package separates CSV loading (`data_io.py`), feature/target
construction (`dataset.py`), and the command-line interface (`__main__.py`).
The original analysis scripts remain the source of the generated Pi files.

```python
from motor import load_dataset
from motor.dataset import RAW_FEATURES, PI_FEATURES

d1t1a, d1t1b, d1t2, d2t1 = load_dataset(max_horizon=5)
X, y = d1t1a
X_raw = X[RAW_FEATURES]
X_pi = X[PI_FEATURES]
```

Each experiment has its own `(X, y)` tuple. No combined series is returned.
The result also exposes named attributes, e.g. `load_dataset().d1t1a`.
Within each pair, the index identifies the original sample and rows remain
in chronological order.

`X` contains nine current operational variables and six selected Pi groups.
`y` contains pointwise fuel consumption at the current instant and each of
the following five seconds. Consumption and Pi groups containing consumption
are excluded from features. Missing features remain NaN; rows without complete
targets are removed. Windows never cross experiment boundaries, and elapsed
time is checked against each horizon. No lag features are added.

Keep these four pairs separate when constructing temporal windows or selecting
training and testing experiments.
`pi_1`, `pi_2`, and `pi_3` duplicate the control variables; `pi_4` includes elapsed
recording time and should be assessed separately when comparing representations.
For a one-second horizon, run `uv run python -m motor --max-horizon 1`.

## Model workflow

Run initial preparation, then load the model dataset:

```bash
uv run prepare-data
uv run model
```

`prepare-data` executes the original `Main_Motor.py` pipeline through
`motor/preprocessing.py`. It generates six CSVs and twenty PNG plots under
`Dados/`, overwriting generated files with the same names. Plotting uses the
headless `Agg` backend unless `MPLBACKEND` is already set.

The dedicated model target currently loads four independent `(X, y)` pairs:

```bash
uv run model
```

Its entry point is `motor/model.py`, where subsequent training and evaluation
stages can be added. To configure the data directory or prediction horizon:

```bash
uv run model --max-horizon 1
uv run model --data-dir /path/to/data --max-horizon 5
```

The command is registered under `[project.scripts]` in `pyproject.toml`.
