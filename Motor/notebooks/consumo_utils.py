"""Leitura e janelas causais compartilhadas pelos notebooks de consumo."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

RAW_SENSORS = ['Velo', 'Acc', 'Brake', 'Gear', 'n', 'Tau', 'Taccm', 'TeLam', 'Team']
HISTORY_SENSORS = ['Velo', 'Acc', 'n', 'Tau', 'Taccm']


def load_data(root):
    """Converte blocos largos em ensaios e junta os Pi por chave temporal."""
    folder = Path(root) / 'Motor' / 'Dados'
    read = lambda name: pd.read_csv(folder / name, sep=';', decimal=',', encoding='latin-1')
    raw = read('Data_set.CSV')
    param = read('Data_set_param.CSV')
    raw.columns = raw.columns.str.strip()
    parts = []
    for experiment in ['D1T1A', 'D1T1B', 'D1T2', 'D2T1']:
        mapping = {
            f'{experiment} - {str(row["Colunas - Parametros"]).strip()}': str(row['Alias']).strip()
            for _, row in param.iterrows()
        }
        trial = raw[list(mapping)].rename(columns=mapping)
        trial = trial.apply(pd.to_numeric, errors='raise').dropna(how='all').copy()
        trial['Experimento'] = experiment
        parts.append(trial)
    data = pd.concat(parts, ignore_index=True).sort_values(['Experimento', 'Temp'])
    pis = read('Dataset_Modelagem_Pi.csv')
    keys = ['Experimento', 'Temp']
    data = data.merge(pis, on=keys, how='left', validate='one_to_one', indicator=True)
    if not data['_merge'].eq('both').all():
        raise ValueError('Há amostras sem correspondência no CSV Pi; execute Main_Motor.py.')
    return data.drop(columns='_merge').replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def forecast_frame(data, horizon=10, history=10, stride=10):
    """Entradas até t; alvo com as h taxas de t+1 a t+h, por ensaio.

    Requer contador em segundos e incremento unitário. Soma retangular:
    cada taxa representa 1 s. Não converte a unidade declarada de Taccm.
    """
    if min(horizon, history, stride) < 1:
        raise ValueError('horizon, history e stride devem ser positivos')
    features = [f'cur_{c}' for c in RAW_SENSORS] + ['cur_Pi_6_log10', 'cur_Pi_8']
    features += [f'{stat}_{c}' for c in HISTORY_SENSORS for stat in ['mean', 'std', 'delta']]
    parts = []
    for experiment, group in data.groupby('Experimento', sort=True):
        g = group.sort_values('Temp').reset_index(drop=True)
        if g.Temp.isna().any() or not np.allclose(g.Temp.diff().dropna(), 1):
            raise ValueError(f'{experiment}: requer intervalo de 1 segundo')
        frame = g[['Temp']].copy()
        frame['Experimento'] = experiment
        frame['feature_start'] = g.Temp - history + 1
        frame['target_start'] = g.Temp + 1
        frame['target_end'] = g.Temp + horizon
        for col in RAW_SENSORS + ['Pi_6_log10', 'Pi_8']:
            frame[f'cur_{col}'] = g[col]
        for col in HISTORY_SENSORS:
            roll = g[col].rolling(history, min_periods=history)
            frame[f'mean_{col}'] = roll.mean()
            frame[f'std_{col}'] = roll.std(ddof=0)
            frame[f'delta_{col}'] = g[col] - g[col].shift(history - 1)
        future = pd.concat([g.Taccm.shift(-i) for i in range(1, horizon + 1)], axis=1)
        valid = future.notna().all(axis=1)
        frame['fuel_future_sum'] = future.sum(axis=1, min_count=horizon)
        frame['fuel_future_mean'] = future.mean(axis=1).where(valid)
        frame = frame.iloc[history - 1::stride].dropna(subset=['fuel_future_sum'])
        parts.append(frame)
    return pd.concat(parts, ignore_index=True), features


def regressor():
    """Configuração fixa, sem procurar hiperparâmetros nos ensaios de teste."""
    return make_pipeline(
        SimpleImputer(strategy='median', add_indicator=True),
        HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, l2_regularization=1,
            early_stopping=False, random_state=42,
        ),
    )


def accelerator_scenario(features, delta, history=10):
    """Altera Acc(t), mantendo t−history+1...t−1 e resumos consistentes."""
    if history < 2:
        raise ValueError('O cenário requer history >= 2')
    result = features.copy()
    current = features['cur_Acc']
    mean = features['mean_Acc']
    variance = features['std_Acc'] ** 2
    new_mean = mean + delta / history
    new_second_moment = variance + mean ** 2 + (2 * current * delta + delta ** 2) / history
    result['cur_Acc'] = current + delta
    result['mean_Acc'] = new_mean
    result['std_Acc'] = np.sqrt(np.maximum(0, new_second_moment - new_mean ** 2))
    result['delta_Acc'] = features['delta_Acc'] + delta
    return result
