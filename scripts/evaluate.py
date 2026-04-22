"""
Time-based holdout evaluation for the Santa Clara daily AQI model.

Trains a RandomForest on all days before the holdout window (same feature recipe
as production), evaluates one-step-ahead targets on the last ``holdout_days`` rows,
and writes metrics to ``metrics.json`` for dashboards or retrain gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Run as ``python scripts/evaluate.py`` from repo root (cwd flexible).
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from feature_engineering import create_time_series_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout MAE / RMSE / R² for county AQI model.")
    parser.add_argument("--csv", default="California_airquality.csv", help="Path to air quality CSV")
    parser.add_argument("--county", default="Santa Clara")
    parser.add_argument("--target", default="DAILY_AQI_VALUE")
    parser.add_argument("--n-lags", type=int, default=7)
    parser.add_argument("--holdout-days", type=int, default=30, help="Last N days used as test set")
    parser.add_argument(
        "--output",
        default="metrics.json",
        help="Where to write metrics (default metrics.json; gitignored by default)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path.resolve()}")
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    county = df[df["COUNTY"] == args.county].copy()
    county[args.target] = pd.to_numeric(county[args.target], errors="coerce")
    county = county.dropna(subset=[args.target, "Date"])
    # Drop non-physical rows that appear in some exports (they destroy daily means).
    county = county[(county[args.target] >= 0) & (county[args.target] <= 500)]

    daily_series = county.groupby("Date", as_index=True)[args.target].mean().sort_index()
    daily = daily_series.to_frame(name=args.target)

    featured = create_time_series_features(daily, args.target, n_lags=args.n_lags)
    if len(featured) <= args.holdout_days + args.n_lags + 2:
        print("Error: Not enough rows for holdout after feature engineering.")
        sys.exit(1)

    split = len(featured) - args.holdout_days
    train_df = featured.iloc[:split]
    test_df = featured.iloc[split:]

    y_train = train_df[args.target]
    X_train = train_df.drop(columns=[args.target])
    y_test = test_df[args.target]
    X_test = test_df.drop(columns=[args.target])

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(mean_squared_error(y_test, pred, squared=False))
    r2 = float(r2_score(y_test, pred))

    payload = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "holdout_days": args.holdout_days,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "target": args.target,
        "county": args.county,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
