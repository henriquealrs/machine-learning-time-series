import pandas as pd

OIL_TEMP_FEATURES = ["oil_temperature", "pi_7"]

def remove_oil_temperatures(X: pd.DataFrame) -> pd.DataFrame:
    X_copy = X.copy()
    for feat in OIL_TEMP_FEATURES:
        X_copy = X_copy.drop(feat, axis=1)

    return X_copy
