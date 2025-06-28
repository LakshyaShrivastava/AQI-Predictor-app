"""
Author: Lakshya Shrivastava

Model Training Script for the AQI Predictor Project.

This script is responsible for training the machine learning model on the entire
historical dataset. It performs the following steps:
1.  Loads the full, updated dataset from the CSV file.
2.  Prepares and cleans the data (filters by county, aggregates daily values).
3.  Engineers time-series features (lags and rolling windows).
4.  Trains a RandomForestRegressor model on all available data.
5.  Saves the trained model to a .joblib file for use by the prediction scripts.

This script is intended to be run periodically (e.g., weekly by a GitHub Action)
to create an updated model that has learned from newly collected data.
"""

# --- Imports ---
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- Configuration ---
CONFIG = {
    "source_csv_path": "California_airquality.csv",
    "output_model_path": "models/model_santa_clara_fire_aware.joblib",
    "county_to_filter": "Santa Clara",
    "target_column": "DAILY_AQI_VALUE",
    "n_lags": 7,
    "model_params": {
        "n_estimators": 100,  # Number of trees in the forest
        "random_state": 42,   # Ensures reproducibility
        "n_jobs": -1          # Use all available CPU cores for faster training
    }
}


# --- 1. Load and Prepare Data ---
print("--- Starting Model Training Process ---")
print(f"Loading dataset from '{CONFIG['source_csv_path']}'...")
try:
    df = pd.read_csv(CONFIG['source_csv_path'], low_memory=False)
except FileNotFoundError:
    print(f"Error: Could not find '{CONFIG['source_csv_path']}'")
    exit()

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Filter for a specific county
county_name = 'Santa Clara'
df_county = df[df['COUNTY'] == county_name].copy()

# Aggregate data by date: there can be multiple monitoring sites per county,
# so we calculate the mean AQI for each day to create a single, clean time series.
df_aqi = df_county.groupby('Date')[CONFIG['target_column']].mean().to_frame()
df_aqi = df_aqi.sort_index()

print(f"✅ Data loaded and prepared for '{CONFIG['county_to_filter']}' County.")
print(f"Data shape after aggregation: {df_aqi.shape}")

# --- 2. Feature Engineering ---

def create_time_series_features(df, target_col, n_lags=7):
    """
    Creates time-series features (lags and rolling stats) from the input data.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime index and a target column.
        target_col (str): The name of the column to create features from.
        n_lags (int): The number of past time steps to use for lag features.

    Returns:
        pd.DataFrame: A new DataFrame with the engineered features and target variable.
    """

    df_featured = df.copy()
    # Create lag features (the value from N days ago).
    for i in range(1, n_lags + 1):
        df_featured[f'{target_col}_lag_{i}'] = df_featured[target_col].shift(i)

    # Create rolling window features (summary stats over a recent period).
    # .shift(1) ensures that the rolling stats for a given day do not include that day's own value.
    rolling_window = df_featured[target_col].shift(1).rolling(window=n_lags)
    df_featured[f'{target_col}_rolling_mean_{n_lags}'] = rolling_window.mean()
    df_featured[f'{target_col}_rolling_std_{n_lags}'] = rolling_window.std()

    # Drop rows with NaN values that were created by the shift/rolling operations.
    df_featured.dropna(inplace=True)
    return df_featured

print("\nEngineering time-series features...")
df_model_data = create_time_series_features(df_aqi, CONFIG['target_column'], CONFIG['n_lags'])

# --- 3. Prepare Data for Model ---
# Separate the features (X) from the target variable (y).
y = df_model_data[CONFIG['target_column']]
X = df_model_data.drop(CONFIG['target_column'], axis=1)

# For this script, we use all available data for training to create the most
# knowledgeable "production" model. There is no train-test split.
X_train = X
y_train = y
print(f"✅ Features prepared. Using {len(X_train)} rows for training.")

# --- 4. Model Training ---

model_fire_aware = RandomForestRegressor(**CONFIG['model_params'])

print(f"\nTraining '{CONFIG['output_model_path']}'...")
# Train the model on the entire feature set.
model_fire_aware.fit(X_train, y_train)
print("✅ Model training complete.")


# --- 5. Save the Trained Model ---
# Serialize the trained model object and save it to a file using joblib.
# This file can then be loaded by other scripts for making predictions.
joblib.dump(model_fire_aware, CONFIG['output_model_path'])
print(f"\n✅ Model has been trained and saved as '{CONFIG['output_model_path']}'.")
