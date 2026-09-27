import pandas as pd
from .data_init import scale_features, split_data

OIL_TEMP_FEATURES = ["oil_temperature", "pi_7"]

def remove_oil_temperatures(X: pd.DataFrame) -> pd.DataFrame:
    X_copy = X.copy()
    for feat in OIL_TEMP_FEATURES:
        X_copy = X_copy.drop(feat, axis=1)

    return X_copy

def train_model_ignore_oil(X: pd.DataFrame, y: pd.DataFrame, metadata: pd.DataFrame):
    X = remove_oil_temperatures(X)
    X_scaled, _ = scale_features(X)
    y_scaled, _ = scale_features(y)
    Xtrain, ytrain, Xtest, ytest = split_data(X_scaled, y_scaled, metadata)

