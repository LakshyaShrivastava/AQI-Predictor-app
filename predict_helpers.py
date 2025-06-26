
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# --- Helper Function for AQI Calculation ---
def pm25_to_aqi(pm25):
    """
    Converts a PM2.5 concentration (in μg/m³) to the US EPA AQI scale.
    """
    if pm25 is None or pm25 < 0:
        return None
    if 0.0 <= pm25 <= 12.0:
        return _calculate_aqi(pm25, 50, 0, 12.0, 0.0)
    elif 12.1 <= pm25 <= 35.4:
        return _calculate_aqi(pm25, 100, 51, 35.4, 12.1)
    elif 35.5 <= pm25 <= 55.4:
        return _calculate_aqi(pm25, 150, 101, 55.4, 35.5)
    elif 55.5 <= pm25 <= 150.4:
        return _calculate_aqi(pm25, 200, 151, 150.4, 55.5)
    elif 150.5 <= pm25 <= 250.4:
        return _calculate_aqi(pm25, 300, 201, 250.4, 150.5)
    elif 250.5 <= pm25 <= 500.4:
        return _calculate_aqi(pm25, 500, 301, 500.4, 250.5)
    else:
        return 500

def _calculate_aqi(Cp, Ih, Il, Ch, Cl):
    """Private helper function for the linear interpolation formula."""
    return round(((Ih - Il) / (Ch - Cl)) * (Cp - Cl) + Il)

# --- Helper Function for OpenWeatherMap API ---

def get_historical_pm25(lat, lon, date_to_fetch, api_key):
    """
    Fetches the historical daily average PM2.5 concentration from OpenWeatherMap.
    """
    start_of_day = date_to_fetch.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    start_timestamp = int(start_of_day.timestamp())
    end_timestamp = int(end_of_day.timestamp())
    
    api_url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
               f"lat={lat}&lon={lon}&start={start_timestamp}&end={end_timestamp}&appid={api_key}")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        if 'list' in data and len(data['list']) > 0:
            daily_pm25_values = [item['components']['pm2_5'] for item in data['list']]
            return np.mean(daily_pm25_values)
        else:
            print(f"Warning: No data returned for {date_to_fetch.strftime('%Y-%m-%d')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {date_to_fetch.strftime('%Y-%m-%d')}: {e}")
        return None

# --- Helper Function for Feature Engineering ---

def create_features_for_prediction(data_window, target_column, n_lags):
    """
    Takes a list/deque of recent data and creates a feature row for the model.
    """
    data = np.array(data_window)
    features = {}
    
    for i in range(n_lags):
        features[f'{target_column}_lag_{i+1}'] = data[n_lags-1-i]
        
    features[f'{target_column}_rolling_mean_7'] = np.mean(data)
    features[f'{target_column}_rolling_std_7'] = np.std(data)
    
    feature_names = [f'{target_column}_lag_{i+1}' for i in range(n_lags)] + \
                    [f'{target_column}_rolling_mean_7', f'{target_column}_rolling_std_7']
    
    return pd.DataFrame([features], columns=feature_names)
