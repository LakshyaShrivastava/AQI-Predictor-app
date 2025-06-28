"""
Author: Lakshya Shrivastava

Helper Functions for the AQI Prediction Project.

This module provides the core, reusable functionalities required by the other scripts:
1.  Conversion from PM2.5 concentration to the US EPA AQI standard.
2.  Fetching historical air pollution data from the OpenWeatherMap API.
3.  Engineering time-series features for the machine learning model.
"""

# --- Imports ---
# Third-party libraries
import requests
import numpy as np
import pandas as pd

# Standard library imports
from datetime import timedelta


# --- Helper Functions for AQI Calculation ---

def pm25_to_aqi(pm25):
    """
    Converts a PM2.5 concentration (in μg/m³) to the US EPA AQI scale.
    This version uses a more robust, contiguous range check.

    Args:
        pm25 (float | None): The measured PM2.5 concentration.

    Returns:
        int | None: The calculated US EPA AQI value, or None if the input is invalid.
    """
    if pm25 <= 12.0:
        return _calculate_aqi(pm25, 50, 0, 12.0, 0.0)
    elif pm25 <= 35.4:
        return _calculate_aqi(pm25, 100, 51, 35.4, 12.1)
    elif pm25 <= 55.4:
        return _calculate_aqi(pm25, 150, 101, 55.4, 35.5)
    elif pm25 <= 150.4:
        return _calculate_aqi(pm25, 200, 151, 150.4, 55.5)
    elif pm25 <= 250.4:
        return _calculate_aqi(pm25, 300, 201, 250.4, 150.5)
    elif pm25 <= 500.4:
        return _calculate_aqi(pm25, 500, 301, 500.4, 250.5)
    else:  # Values above 500.4 are considered beyond the standard scale
        return 501  # Returning 501+ for values off the chart is common practice

def _calculate_aqi(Cp, Ih, Il, Ch, Cl):
    """
    Private helper function for the linear interpolation formula.

    Args:
        Cp (float): The input PM2.5 concentration.
        Ih (int): The AQI value corresponding to the high end of the category.
        Il (int): The AQI value corresponding to the low end of the category.
        Ch (float): The PM2.5 concentration for the high end of the category.
        Cl (float): The PM2.5 concentration for the low end of the category.

    Returns:
        int: The final, rounded AQI value
    """
    return round(((Ih - Il) / (Ch - Cl)) * (Cp - Cl) + Il)

# --- Helper Function for OpenWeatherMap API ---

def get_historical_pm25(lat, lon, date_to_fetch, api_key):
    """
    Fetches the historical daily average PM2.5 concentration from OpenWeatherMap.

    Args:
        lat (float): Latitude of the desired location.
        lon (float): Longitude of the desired location.
        date_to_fetch (datetime): The specific date for which to fetch data.
        api_key (str): Your valid OpenWeatherMap API key.

    Returns:
        float | None: The daily average PM2.5 concentration, or None if the API
                      call fails or returns no data.
    """
    # The API requires start and end times as Unix timestamps.
    start_of_day = date_to_fetch.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    start_timestamp = int(start_of_day.timestamp())
    end_timestamp = int(end_of_day.timestamp())
    
    # Construct the API request URL with the required parameters.
    api_url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
               f"lat={lat}&lon={lon}&start={start_timestamp}&end={end_timestamp}&appid={api_key}")
    
    try:
        response = requests.get(api_url)
        # Raise an exception for bad HTTP status codes (like 401 Unauthorized or 404 Not Found).
        response.raise_for_status() 
        data = response.json()
        
         # OWM provides hourly data; we average it to get a single daily value.
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
    Creates a single-row DataFrame of features for the ML model from a window of recent data.

    This function must generate the exact same features that the model was trained on.

    Args:
        data_window (Iterable): A list, deque, or array of the last N days of AQI values.
        target_column (str): The name of the target variable (e.g., 'DAILY_AQI_VALUE').
        n_lags (int): The number of lag features to create (should match the window size).

    Returns:
        pd.DataFrame: A single-row DataFrame ready to be fed into the model's .predict() method.
    """
    data = np.array(data_window)
    features = {}
    
    # Create lag features (e.g., AQI_lag_1, AQI_lag_2, ...)
    for i in range(n_lags):
        # Access the array from the end (most recent) to the start (oldest).
        features[f'{target_column}_lag_{i+1}'] = data[n_lags-1-i]

    # Create rolling window features that summarize the period.
    features[f'{target_column}_rolling_mean_7'] = np.mean(data)
    features[f'{target_column}_rolling_std_7'] = np.std(data)
    
    # Define the exact column order to match the model's training.
    feature_names = [f'{target_column}_lag_{i+1}' for i in range(n_lags)] + \
                    [f'{target_column}_rolling_mean_7', f'{target_column}_rolling_std_7']
    
    return pd.DataFrame([features], columns=feature_names)
