"""
Author: Lakshya Shrivastava

Automated Data Collection and Prediction Logging Script.

This script serves two primary functions in the MLOps pipeline:
1.  **Performance Logging:** It generates a prediction for the previous day (yesterday)
    based on the data that was available at the time and saves this prediction to a log file.
    This allows us to compare past predictions with actual outcomes.
2.  **Data Collection:** It fetches the actual, measured AQI for the previous day and
    appends it to the main historical dataset, allowing the model to be retrained
    on new data over time.

This script is designed to be run once a day automatically (e.g., via a GitHub Action).
"""

# --- Imports ---
# Standard library imports
import os
from collections import deque
from datetime import datetime, timedelta

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Local application/library specific imports
from predict_helpers import (create_features_for_prediction,
                             get_historical_pm25, pm25_to_aqi)

# --- 1. Define Constants & Load Resources ---

# File paths
DATASET_FILENAME = 'California_airquality.csv'
PREDICTION_LOG_FILENAME = 'prediction_log.csv'
MODEL_FILENAME = 'models/model_santa_clara_fire_aware.joblib'

# Location and Model parameters
COUNTY_NAME = 'Santa Clara'
STATE_NAME = 'California'
LATITUDE = 37.4323
LONGITUDE = -121.8996
N_LAGS = 7
TARGET_COLUMN = 'DAILY_AQI_VALUE'

# Load the API key from environment variables for security
API_KEY = os.getenv("OWM_API_KEY")
if not API_KEY:
    print("❌ Error: OWM_API_KEY environment variable not set.")
    exit()

# Load the pre-trained machine learning model
try:
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ Model '{MODEL_FILENAME}' loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_FILENAME}' not found.")
    exit()


# --- 2. Generate and Log Yesterday's Prediction ---
print("\n--- Generating and Logging Prediction for Yesterday ---")

# Define "yesterday" as the target date for both prediction and data collection.
yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
yesterday_str_format = yesterday.strftime('%Y-%m-%d')

# To predict for yesterday, we need the 7 days of data PRIOR to it.
# We loop from 8 days ago up to 2 days ago to build the feature set.
prediction_input_data = []
print(f"Fetching data from the 7 days prior to {yesterday_str_format} to generate a prediction...")
for i in range(N_LAGS, 0, -1):  # Corrected loop: Fetches data from day-7 up to day-1 before our target `yesterday`.
    date_to_fetch = yesterday - timedelta(days=i)
    pm25_val = get_historical_pm25(LATITUDE, LONGITUDE, date_to_fetch, API_KEY)
    aqi_val = pm25_to_aqi(pm25_val)
    if aqi_val is not None:
        prediction_input_data.append(aqi_val)
    else:  # Simple fallback for missing API data
        prediction_input_data.append(50) if not prediction_input_data else prediction_input_data.append(prediction_input_data[-1])

# Ensure we have a full 7 days of data before making a prediction
if len(prediction_input_data) == N_LAGS:
    # Create features from the historical data
    features = create_features_for_prediction(prediction_input_data, TARGET_COLUMN, N_LAGS)
    # Generate the prediction for what "yesterday" would have been
    predicted_aqi = model.predict(features)[0]
    print(f"✅ Model's prediction for {yesterday_str_format}: {int(predicted_aqi)}")

    # Load prediction log, or create it if it doesn't exist
    try:
        log_df = pd.read_csv(PREDICTION_LOG_FILENAME, parse_dates=['Date'])
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['Date', 'Predicted_AQI'])
    
    # Create new log entry and append it, but only if it's not a duplicate
    new_log_entry = pd.DataFrame([{'Date': yesterday, 'Predicted_AQI': predicted_aqi}])
    if yesterday not in log_df['Date'].values:
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        # Ensure date format is consistent upon saving
        log_df['Date'] = log_df['Date'].dt.strftime('%Y-%m-%d')
        log_df.to_csv(PREDICTION_LOG_FILENAME, index=False)
        print(f"✅ Prediction for {yesterday_str_format} saved to log.")
    else:
        print(f"⚠️ Prediction for {yesterday_str_format} already exists in log.")
else:
    print("❌ Could not gather enough historical data to create a prediction for yesterday.")


# --- 3. Fetch and Append Yesterday's ACTUAL Data ---
print(f"\n--- Fetching and Appending Actual Data for {yesterday_str_format} ---")
pm25_actual = get_historical_pm25(LATITUDE, LONGITUDE, yesterday, API_KEY)
actual_aqi = pm25_to_aqi(pm25_actual)

if actual_aqi:
    print(f"✅ Actual measured AQI for {yesterday_str_format} was {actual_aqi}.")
    try:
        # Load the main dataset, ensuring dates are handled correctly
        main_df = pd.read_csv(DATASET_FILENAME, low_memory=False)
        main_df['Date'] = pd.to_datetime(main_df['Date'], errors='coerce')
        main_df.dropna(subset=['Date'], inplace=True)

        # Check for duplicates before appending
        if yesterday not in main_df['Date'].values:
            # Create a new row that matches the structure of the main CSV
            new_actual_row = pd.DataFrame([{
                'Date': yesterday, 'COUNTY': COUNTY_NAME, 'STATE': STATE_NAME,
                'DAILY_AQI_VALUE': actual_aqi, 'SITE_LATITUDE': LATITUDE, 'SITE_LONGITUDE': LONGITUDE,
                'Source': 'OpenWeatherMap API', 'Site ID': np.nan, 'POC': np.nan,
                'Daily Mean PM2.5 Concentration': pm25_actual, 'UNITS': 'ug/m3',
                'DAILY_OBS_COUNT': np.nan, 'PERCENT_COMPLETE': np.nan,
                'AQS_PARAMETER_CODE': 88101, 'AQS_PARAMETER_DESC': 'PM2.5 - Local Conditions',
                'CBSA_CODE': np.nan, 'CBSA_NAME': np.nan, 'STATE_CODE': np.nan,
                'COUNTY_CODE': np.nan, 'Site Name': 'Aggregated by County'
            }])
            
            # Append the new row and save the updated dataset
            df_updated = pd.concat([main_df, new_actual_row], ignore_index=True)
            df_updated['Date'] = df_updated['Date'].dt.strftime('%Y-%m-%d')
            df_updated.to_csv(DATASET_FILENAME, index=False)
            print(f"✅ Actual data for {yesterday_str_format} appended to main dataset.")
        else:
             print("⚠️ Actual data for yesterday already exists in main dataset.")
    except Exception as e:
        print(f"❌ Error updating main dataset: {e}")
else:
    print(f"❌ Could not fetch actual AQI for {yesterday_str_format}.")