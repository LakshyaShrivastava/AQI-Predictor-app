import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# --- Load and Prepare Data ---

# Load the dataset
try:
    df = pd.read_csv('California_airquality.csv')
except FileNotFoundError:
    print("Error: could not fine 'California_airquality.csv'")
    exit()

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Filter for a specific county
county_name = 'Santa Clara'
df_county = df[df['COUNTY'] == county_name].copy()

# There can be multiple sites per county. We need to create a single daily
# value for the county by averaging the AQI across all of its sites.
target_column = 'DAILY_AQI_VALUE'

# Group by date and calculate the mean AQI for that day.
# .to_frame() converts the resulting Series back into a DataFrame.
df_aqi = df_county.groupby('Date')[target_column].mean().to_frame()

# Ensure the new DataFrame is sorted by date
df_aqi = df_aqi.sort_index()

print(f"Loaded and filtered data for {county_name} County.")
print("Data shape:", df_aqi.shape)
print("First 5 rows:")
print(df_aqi.head())

# --- Feature Engineering ---

def create_time_series_features(df, target_col, n_lags=7):
    df_featured = df.copy()
    for i in range(1, n_lags + 1):
        df_featured[f'{target_col}_lag_{i}'] = df_featured[target_col].shift(i)
    df_featured[f'{target_col}_rolling_mean_7'] = df_featured[target_col].shift(1).rolling(window=7).mean()
    df_featured[f'{target_col}_rolling_std_7'] = df_featured[target_col].shift(1).rolling(window=7).std()
    df_featured.dropna(inplace=True)
    return df_featured

df_model_data = create_time_series_features(df_aqi, target_column, n_lags=7)

y = df_model_data[target_column]
X = df_model_data.drop(target_column, axis=1)

# --- Chronological Train-Test Split ---
X_train = X
y_train = y
print(f"\nUsing the entire dataset for training.")
print(f"Training data shape: {X_train.shape}")

# --- Model Training ---

model_fire_aware = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining 'Fire-Aware' Model...")
model_fire_aware.fit(X_train, y_train)
print("Model training complete.")

# --- Save the Trained Model ---

model_filename = 'models/model_santa_clara_fire_aware.joblib'
joblib.dump(model_fire_aware, model_filename)
print(f"\nModel has been trained and saved as '{model_filename}'.")
