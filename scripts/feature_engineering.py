"""
Shared time-series feature definitions for training and inference.

Training and prediction must produce identical feature names and values for the
same history window; this module is the single source of truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_columns_ordered(target_column: str, n_lags: int) -> list[str]:
    """Column order expected by trained RandomForest models."""
    return (
        [f"{target_column}_lag_{i + 1}" for i in range(n_lags)]
        + [
            f"{target_column}_rolling_mean_{n_lags}",
            f"{target_column}_rolling_std_{n_lags}",
        ]
    )


def create_time_series_features(
    df: pd.DataFrame, target_col: str, n_lags: int = 7
) -> pd.DataFrame:
    """
    Build lag and rolling features from a DataFrame with a DateTimeIndex and target_col.

    Rolling mean/std use the same window as training: prior n_lags days only (shift(1)
    so the forecast day is not included). std uses pandas default (ddof=1) to match
    inference which mirrors that statistic on the lag window values.
    """
    df_featured = df.copy()
    for i in range(1, n_lags + 1):
        df_featured[f"{target_col}_lag_{i}"] = df_featured[target_col].shift(i)

    rolling_window = df_featured[target_col].shift(1).rolling(window=n_lags)
    df_featured[f"{target_col}_rolling_mean_{n_lags}"] = rolling_window.mean()
    df_featured[f"{target_col}_rolling_std_{n_lags}"] = rolling_window.std()

    df_featured.dropna(inplace=True)
    return df_featured


def create_features_for_prediction(
    data_window, target_column: str, n_lags: int
) -> pd.DataFrame:
    """
    Single-row feature vector from the last n_lags daily AQI values (oldest to newest
    in the iterable order used historically: the window is the same ordering as
    ``last_7_days_aqi`` in the Streamlit app — index 0 oldest, index -1 most recent).
    """
    data = np.asarray(data_window, dtype=float)
    if data.size != n_lags:
        raise ValueError(f"data_window must have length {n_lags}, got {data.size}")

    features: dict[str, float] = {}
    for i in range(n_lags):
        features[f"{target_column}_lag_{i + 1}"] = float(data[n_lags - 1 - i])

    # Match pandas Rolling.std(ddof=1) on the same n_lags values.
    features[f"{target_column}_rolling_mean_{n_lags}"] = float(np.mean(data))
    features[f"{target_column}_rolling_std_{n_lags}"] = float(np.std(data, ddof=1))

    cols = feature_columns_ordered(target_column, n_lags)
    return pd.DataFrame([features], columns=cols)
