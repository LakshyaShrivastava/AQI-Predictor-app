import pandas as dp
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from predict_helpers import get_historical_pm25, pm25_to_aqi

DATASET_FILENAME = 'California_airquality.csv'

COUNTY_NAME = 'Santa Clara'
STATE_NAME = 'California'

# Can use OWM api to get these based on name
LATITUDE = 37.4323 
LONGITUDE = -121.8996

API_KEY = os.environ.get("OWM_API_KEY")

if not API_KEY:
    print("❌ Error: OWM_API_KEY environment variable not set.")
    exit()

yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
print(f"Attempting to fetch data for {yesterday.strftime('%Y-%m-%d')}...")

pm25_val = get_historical_pm25(LATITUDE, LONGITUDE, yesterday, API_KEY)
yesterday_aqi = pm25_to_aqi(pm25_val)

if yesterday_aqi is None:
    print(f"❌ Could not retrieve valid AQI data for {yesterday.strftime('%Y-%m-%d')}. Exiting.")
    exit()

print(f"✅ Successfully fetched data: AQI for {yesterday.strftime('%Y-%m-%d')} was {yesterday_aqi}.")

# --- 3. Load and Clean Existing Dataset ---
try:
    df = pd.read_csv(DATASET_FILENAME, low_memory=False)
    print(f"Loaded existing dataset: '{DATASET_FILENAME}' with {len(df)} rows.")
except FileNotFoundError:
    print(f"❌ Error: Could not find the dataset file '{DATASET_FILENAME}'.")
    exit()

# --- THIS IS THE FIX ---
# Force the 'Date' column to be a datetime type.
# errors='coerce' will turn any unparseable dates into 'NaT' (Not a Time) instead of crashing.
print("Normalizing date column and cleaning data...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Now, we drop any rows where the date could not be parsed. These rows are unusable.
initial_rows = len(df)
df.dropna(subset=['Date'], inplace=True)
cleaned_rows = len(df)
if initial_rows > cleaned_rows:
    print(f"Removed {initial_rows - cleaned_rows} rows with invalid dates.")
# --- END OF FIX ---

# --- 4. Check for Duplicates and Append New Data ---
if yesterday in df['Date'].values:
    print(f"⚠️ Data for {yesterday.strftime('%Y-%m-%d')} already exists in the dataset. Exiting to prevent duplicates.")
    exit()

new_row_data = {
    'Date': yesterday, 'COUNTY': COUNTY_NAME, 'STATE': STATE_NAME,
    'DAILY_AQI_VALUE': yesterday_aqi, 'SITE_LATITUDE': LATITUDE, 'SITE_LONGITUDE': LONGITUDE,
    'Source': 'OpenWeatherMap API', 'Site ID': np.nan, 'POC': np.nan,
    'Daily Mean PM2.5 Concentration': pm25_val, 'UNITS': 'ug/m3', 'DAILY_OBS_COUNT': np.nan,
    'PERCENT_COMPLETE': np.nan, 'AQS_PARAMETER_CODE': 88101,
    'AQS_PARAMETER_DESC': 'PM2.5 - Local Conditions', 'CBSA_CODE': np.nan, 'CBSA_NAME': np.nan,
    'STATE_CODE': np.nan, 'COUNTY_CODE': np.nan, 'Site Name': 'Aggregated by County'
}
new_row = pd.DataFrame([new_row_data])
df_updated = pd.concat([df, new_row], ignore_index=True)
print(f"Appending new row for {yesterday.strftime('%Y-%m-%d')}. New dataset will have {len(df_updated)} rows.")

# --- 5. Save Updated Dataset ---
# Now this line will work because the 'Date' column is a clean datetime type.
df_updated['Date'] = df_updated['Date'].dt.strftime('%Y-%m-%d')
df_updated.to_csv(DATASET_FILENAME, index=False)

print(f"✅ Successfully updated '{DATASET_FILENAME}' with new data.")
