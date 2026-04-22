"""Contract tests: training-time features must match inference-time features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.feature_engineering import (
    create_features_for_prediction,
    create_time_series_features,
    feature_columns_ordered,
)


def test_create_features_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        create_features_for_prediction([1, 2, 3], "DAILY_AQI_VALUE", 7)


def test_training_row_matches_inference_row_synthetic_series():
    """Last training feature row for day T equals create_features on [y_{T-7},...,y_{T-1}]."""
    target = "DAILY_AQI_VALUE"
    n_lags = 7
    rng = np.random.default_rng(42)
    values = rng.integers(20, 120, size=30).astype(float)

    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    df = pd.DataFrame({target: values}, index=idx)

    featured = create_time_series_features(df, target, n_lags=n_lags)
    assert not featured.empty

    # Use an interior row so the window is fully inside the synthetic series.
    row_date = featured.index[15]
    train_row = featured.loc[row_date]
    window = [
        float(df.loc[row_date - pd.Timedelta(days=k), target])
        for k in range(n_lags, 0, -1)
    ]
    assert len(window) == n_lags
    assert window[0] == df.loc[row_date - pd.Timedelta(days=7), target]
    assert window[-1] == df.loc[row_date - pd.Timedelta(days=1), target]

    infer_df = create_features_for_prediction(window, target, n_lags)
    cols = feature_columns_ordered(target, n_lags)

    for c in cols:
        assert infer_df.loc[0, c] == pytest.approx(float(train_row[c]), rel=1e-9, abs=1e-6)


def test_feature_column_order_stable():
    cols = feature_columns_ordered("DAILY_AQI_VALUE", 7)
    assert cols[0] == "DAILY_AQI_VALUE_lag_1"
    assert cols[6] == "DAILY_AQI_VALUE_lag_7"
    assert cols[7] == "DAILY_AQI_VALUE_rolling_mean_7"
    assert cols[8] == "DAILY_AQI_VALUE_rolling_std_7"
