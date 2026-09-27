"""Protege a separação entre entradas presentes e respostas futuras."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Motor' / 'notebooks'))
from consumo_utils import forecast_frame, RAW_SENSORS


def series_fixture():
    parts = []
    for name, offset in [('A', 0), ('B', 100)]:
        frame = pd.DataFrame({c: np.arange(12, dtype=float) + offset for c in RAW_SENSORS})
        frame['Experimento'] = name
        frame['Temp'] = np.arange(12, dtype=float)
        frame['Pi_6_log10'] = 1.0
        frame['Pi_8'] = .8
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_target_excludes_present_and_never_crosses_trials():
    result, features = forecast_frame(series_fixture(), horizon=2, history=3, stride=1)
    a = result.query("Experimento == 'A' and Temp == 2").iloc[0]
    b = result.query("Experimento == 'B' and Temp == 2").iloc[0]
    assert a['fuel_future_sum'] == 7  # t=3 + t=4; não inclui t=2
    assert a['fuel_future_mean'] == 3.5
    assert b['fuel_future_sum'] == 207
    assert a['mean_Taccm'] == 1
    assert a['feature_start'] == 0 and a['target_start'] == 3 and a['target_end'] == 4
    assert result.groupby('Experimento')['Temp'].max().eq(9).all()
    assert not any('future' in c or c == 'Temp' for c in features)


def test_future_change_cannot_change_current_features():
    original = series_fixture()
    mutated = original.copy()
    mutated.loc[(mutated.Experimento == 'A') & (mutated.Temp > 2), RAW_SENSORS] = 999
    a, features = forecast_frame(original, horizon=2, history=3, stride=1)
    b, _ = forecast_frame(mutated, horizon=2, history=3, stride=1)
    pd.testing.assert_series_equal(a.loc[0, features], b.loc[0, features])
    assert a.loc[0, 'fuel_future_sum'] != b.loc[0, 'fuel_future_sum']


def test_irregular_time_is_rejected_instead_of_called_seconds():
    frame = series_fixture()
    frame.loc[frame.index[5], 'Temp'] = 5.5
    with pytest.raises(ValueError, match='1 segundo'):
        forecast_frame(frame, horizon=2, history=3, stride=1)


def test_missing_future_is_not_a_partial_sum():
    frame = series_fixture()
    frame.loc[(frame.Experimento == 'A') & (frame.Temp == 4), 'Taccm'] = np.nan
    result, _ = forecast_frame(frame, horizon=2, history=3, stride=1)
    assert result.query("Experimento == 'A' and Temp == 2").empty


def test_accelerator_scenario_preserves_consistency_of_rolling_features():
    from consumo_utils import accelerator_scenario
    original = series_fixture()
    changed = original.copy()
    changed.loc[(changed.Experimento == 'A') & (changed.Temp == 4), 'Acc'] += 2
    before, features = forecast_frame(original, horizon=2, history=3, stride=1)
    after, _ = forecast_frame(changed, horizon=2, history=3, stride=1)
    original_row = before.query("Experimento == 'A' and Temp == 4")[features]
    expected = after.query("Experimento == 'A' and Temp == 4")[features]
    actual = accelerator_scenario(original_row, delta=2, history=3)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert original_row.iloc[0]['cur_Acc'] == 4  # não modifica a base
