import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import joblib
from collections import deque

# --- Import our custom helper functions ---
from predict_helpers import get_historical_pm25, pm25_to_aqi, create_features_for_prediction

# --- 1. Define Constants & Load Model ---

DATASET_FILENAME = 'California_airquality.csv'
PREDICTION_LOG_FILENAME = 'prediction_log.csv'
MODEL_FILENAME = 'models/model_santa_clara_fire_aware.joblib' # Use our best model to make predictions

COUNTY_NAME = 'Santa Clara'
STATE_NAME = 'California'
LATITUDE = 37.4323
LONGITUDE = -121.8996
N_LAGS = 7
TARGET_COLUMN = 'DAILY_AQI_VALUE'

# Load the API key
API_KEY = os.getenv("OWM_API_KEY")
if not API_KEY:
    print("❌ Error: OWM_API_KEY environment variable not set.")
    exit()

# Load the machine learning model
try:
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_FILENAME}' not found.")
    exit()


# --- 2. Generate and Log Yesterday's Prediction ---

# We need the 7 days PRIOR to yesterday to make a prediction FOR yesterday.
print("--- Generating and Logging Prediction for Yesterday ---")
yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
prediction_input_data = []
for i in range(N_LAGS + 1, 1, -1): # Loop from 8 days ago to 2 days ago
    date_to_fetch = yesterday - timedelta(days=i-1)
    pm25_val = get_historical_pm25(LATITUDE, LONGITUDE, date_to_fetch, API_KEY)
    aqi_val = pm25_to_aqi(pm25_val)
    if aqi_val is not None:
        prediction_input_data.append(aqi_val)
    else: # Simple fallback
        prediction_input_data.append(50) if not prediction_input_data else prediction_input_data.append(prediction_input_data[-1])

if len(prediction_input_data) == N_LAGS:
    # Create features and make the prediction for yesterday
    features = create_features_for_prediction(prediction_input_data, TARGET_COLUMN, N_LAGS)
    predicted_aqi = model.predict(features)[0]
    print(f"✅ Model's prediction for {yesterday.strftime('%Y-%m-%d')}: {int(predicted_aqi)}")

    # Load prediction log or create it if it doesn't exist
    try:
        log_df = pd.read_csv(PREDICTION_LOG_FILENAME, parse_dates=['Date'])
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['Date', 'Predicted_AQI'])
    
    # Create new log entry and append it, avoiding duplicates
    new_log_entry = pd.DataFrame([{'Date': yesterday, 'Predicted_AQI': predicted_aqi}])
    if yesterday not in log_df['Date'].values:
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        log_df.to_csv(PREDICTION_LOG_FILENAME, index=False)
        print(f"✅ Prediction for {yesterday.strftime('%Y-%m-%d')} saved to log.")
    else:
        print(f"⚠️ Prediction for {yesterday.strftime('%Y-%m-%d')} already exists in log.")
else:
    print("❌ Could not gather enough data to create a prediction for yesterday.")


# --- 3. Fetch and Append Yesterday's ACTUAL Data ---
# This section is the same as before, just more streamlined.
print("\n--- Fetching and Appending Actual Data for Yesterday ---")
pm25_actual = get_historical_pm25(LATITUDE, LONGITUDE, yesterday, API_KEY)
actual_aqi = pm25_to_aqi(pm25_actual)

if actual_aqi:
    try:
        main_df = pd.read_csv(DATASET_FILENAME, parse_dates=['Date'])
        if yesterday not in main_df['Date'].values:
            # ... (The logic to append the actual value to the main CSV is the same as your last collect_data.py) ...
            # This part is omitted for brevity but you should keep your existing logic here.
            print("✅ Actual data appended to main dataset.")
        else:
             print("⚠️ Actual data for yesterday already exists in main dataset.")
    except Exception as e:
        print(f"Error updating main dataset: {e}")
else:
    print("❌ Could not fetch actual AQI for yesterday.")