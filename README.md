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

X, y, metadata = load_dataset(max_horizon=5)
X_raw = X[RAW_FEATURES]
X_pi = X[PI_FEATURES]
```

The returned tables combine all four experiments with aligned row indices.
`metadata` identifies each row's `experiment`, original `sample`, and
`time_seconds`. Target windows are constructed within each experiment before
the rows are combined.

`X` contains nine current operational variables and six selected Pi groups.
`y` contains pointwise fuel consumption at the current instant and each of
the following five seconds. Consumption and Pi groups containing consumption
are excluded from features. Missing features remain NaN; rows without complete
targets are removed. Windows never cross experiment boundaries, and elapsed
time is checked against each horizon. No lag features are added.

Use `metadata["experiment"]` to select training and testing experiments.
Any additional lag or rolling features must also respect experiment boundaries.
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

The model target trains histogram gradient boosting, evaluates each horizon,
and saves the estimator, predictions, metrics, split membership, settings,
and observed-versus-predicted graphs. Its default policy removes oil temperature
and `pi_7`, using all four experiments:

```bash
uv run model
```

To configure the data directory, horizon, or oil policy:

```bash
uv run model --max-horizon 1
uv run model --data-dir /path/to/data --max-horizon 5
uv run model --oil-policy with_oil
```

The command is registered under `[project.scripts]` in `pyproject.toml`.

`with_oil` retains oil-related inputs and filters out rows without those
measurements. `experiment` splitting holds out the smallest remaining nonempty
experiment: D2T1 for `ignore_oil`, D1T1B for `with_oil` with the current data.
Other split strategies can be added in `motor/data_init.py`; unknown names
raise an error. Different oil policies can hold out different experiments,
so their scores are not a controlled comparison of oil temperature's effect.

Each execution creates a new folder:

```text
outputs/<model>/<split_type>/<oil_policy>/<timestamp>_<run_id>/
    model.joblib
    settings.json
    metrics.csv
    predictions.csv
    split.csv
    plots/<test_experiment>.png
    plots/<test_experiment>_residuals.png
    plots/mse_by_horizon.png
```

Output files are ignored by Git. `settings.json` records model parameters,
features, horizons, train/test runs, excluded-row counts, package versions,
training time, and a fingerprint of the input tables. `predictions.csv`
includes the original row, forecast origin, target time, observations,
predictions, and residuals. Metrics and plots use the original consumption
scale; its physical unit still needs confirmation.

`metrics.csv` includes MAE, MSE, RMSE, and R² per horizon.
`plots/mse_by_horizon.png` shows test MSE from t0 through the final horizon,
with additional curves per experiment when several experiments are tested.
MSE is averaged across test samples and expressed in squared source units;
`predictions.csv` also stores each prediction's squared error.

Residual plots show instantaneous signed error (`observed - predicted`) over
target time, with one panel per horizon and a dashed zero reference. Positive
values indicate underestimation; negative values indicate overestimation.
Panels use the same symmetric error scale within each test experiment, and
lines break at gaps in the series.

Model-specific construction belongs in `motor/estimators.py`; add a builder
there and register it in `MODEL_BUILDERS` to reuse training and reporting.
Builders can return pipelines that fit preprocessing on training data only.
Histogram gradient boosting needs no scaling and directly accepts NaNs.
Training is shared in `motor/training.py`, reporting in `motor/reporting.py`.

```python
from motor import load_dataset
from motor.model_no_oil_temp import train_model_ignore_oil

X, y, metadata = load_dataset()
result = train_model_ignore_oil(X, y, metadata, split_type="experiment")
print(result.metrics)
print(result.output_dir)
```

To apply a shared scaler directly:

```python
from motor.data_init import scale_features

X_scaled, scaler = scale_features(X)
```

Scaling preserves missing values and does not modify `y` or `metadata`.
For held-out evaluation, fit the scaler on training rows only and apply
`scaler.transform(X_test)` to test rows.
