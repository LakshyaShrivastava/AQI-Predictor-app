"""
Author: Lakshya Shrivastava

This is the main script for the Streamlit web application that serves as the user interface
for the AQI Prediction project.

It handles loading the models, fetching live data, generating forecasts, and displaying
the results in an interactive dashboard with multiple tabs.
"""

# --- Imports ---
# Standard library imports
import os
from collections import deque
from datetime import datetime, timedelta

# Third-party imports
import joblib
import pandas as pd
import streamlit as st

# Custom helper function imports
from scripts.predict_helpers import get_historical_pm25, pm25_to_aqi, create_features_for_prediction
from scripts.ui_helpers import get_aqi_display_style

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Forecaster",
    page_icon="💨",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_resource
def load_model(model_path):
    """
    Loads a machine learning model from a .joblib file.

    Uses Streamlit's cache_resource to ensure the model is loaded from disk
    only once, which significantly improves performance.

    Args:
        model_path (str): The file path to the .joblib model file.

    Returns:
        object: The loaded scikit-learn model object, or None if not found.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def fetch_recent_data(lat, lon, api_key, n_days):
    '''
    Fetches the last N days of historical AQI data from the OpenWeatherMap API.

    Uses Streamlit's cache_data to prevent re-fetching data from the API on every
    page interaction, making the app much faster. The cache will only clear
    if the input arguments to this function change.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        api_key (str): Your OpenWeatherMap API key.
        n_days (int): The number of past days of data to fetch.

    Returns:
        list: A list of the last N days of calculated US EPA AQI values.

    '''

    print("Fetching recent data from API...")
    recent_aqi = []
    today = datetime.now()
    for i in range(n_days, 0, -1):
        date_to_fetch = today - timedelta(days=i)
        pm25_val = get_historical_pm25(lat, lon, date_to_fetch, api_key)
        aqi_val = pm25_to_aqi(pm25_val)
        if aqi_val is not None:
            recent_aqi.append(aqi_val)
        else:
            recent_aqi.append(50) if not recent_aqi else recent_aqi.append(recent_aqi[-1])
    return recent_aqi

# --- Main Application Logic ---
def run_app():
    """
    The main function that orchestrates the Streamlit application.
    It handles data loading, forecasting, and rendering all UI components.
    """

    # --- Sidebar for User input ---
    st.sidebar.header("Model Settings")
    model_choice_name = st.sidebar.selectbox(
        "Choose a Prediction Model:",
        ('model_santa_clara_fire_aware.joblib', 'model_2020.joblib')
    )

    model_choice_path = os.path.join("models", model_choice_name)

    # --- Setup and Initialization ---
    TARGET_COLUMN = 'DAILY_AQI_VALUE'
    N_LAGS = 7
    # Securely get the API key from Streamlit Secrets or local environment variables
    try:
        API_KEY = st.secrets.get("OWM_API_KEY")
    except:
        API_KEY = os.getenv("OWM_API_KEY")

    if not API_KEY:
        st.error("OWM_API_KEY not found. Please add it to your Streamlit Secrets.")
        st.stop()

    # --- Model and Data Loading ---
    model = load_model(model_choice_path)
    if model is None:
        st.error(f"Model file '{model_choice_name}' not found.")
        st.stop()
    
    # Display the last time the selected model file was trained/modified
    last_trained_timestamp = os.path.getmtime(model_choice_path)
    last_trained_date = datetime.fromtimestamp(last_trained_timestamp)

   # Fetch the last 7 days of data to use as input for the forecast
    last_7_days_aqi = fetch_recent_data(37.4323, -121.8996, API_KEY, N_LAGS)

    # --- Forecasting Logic ---
    # Use a deque for an efficient sliding window of data
    current_window = deque(last_7_days_aqi, maxlen=N_LAGS)
    future_predictions = []
    # Loop 7 times to generate a 7-day forecast
    for _ in range(7):
        # Create features from the current window of 7 days
        input_features = create_features_for_prediction(current_window, TARGET_COLUMN, N_LAGS)
        # Predict the next day's AQI
        predicted_aqi = model.predict(input_features)[0]
        # Add the prediction to our forecast list
        future_predictions.append(predicted_aqi)
        # Update the window: remove the oldest day and add the new prediction
        current_window.append(predicted_aqi)

    # --- Data Preparation for UI ---
    # Create a DataFrame for easy plotting and display
    start_date = datetime.now() + timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, periods=7)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted AQI': future_predictions}).set_index('Date')

    # --- Main Page Layout and Tabs ---
    st.title("💨 Santa Clara County AQI Forecaster")
    st.markdown("An adaptive machine learning model for daily Air Quality Index (AQI) prediction.")

    tab_forecast, tab_performance, tab_legend, tab_about = st.tabs(["📈 Forecast", "📊 Model Performance", "🎨 AQI Legend", "🤖 About the Model"])

    # --- Forecast Tab ---
    with tab_forecast:
        st.header(f"7-Day Forecast using `{model_choice_name}`")
        st.info(f"Prediction based on the last 7 days of actual AQI: `{[int(v) for v in last_7_days_aqi]}`")

        cols = st.columns(7)
        for i, col in enumerate(cols):
            with col:
                date = forecast_df.index[i]
                aqi_value = forecast_df['Predicted AQI'][i]
                style = get_aqi_display_style(aqi_value)
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 10px; text-align: center; {style}">
                    <div style="font-size: 1em; font-weight: bold;">{date.strftime('%A')}</div>
                    <div style="font-size: 0.8em;">{date.strftime('%b %d')}</div>
                    <div style="font-size: 2.5em; font-weight: bold;">{int(aqi_value)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.subheader("Forecast Chart")
        st.line_chart(forecast_df['Predicted AQI'])

    # --- Model Performance Tab ---
    with tab_performance:
        st.header("Historical Model Performance")
        st.write("This chart compares the model's past daily predictions against the actual measured AQI for that day.")
        try:
            # Load the log of past predictions and the main dataset of actuals
            predictions_df = pd.read_csv('prediction_log.csv', parse_dates=['Date'])
            
            # Aggregate the actuals data by day to ensure a clean time series
            source_df = pd.read_csv('California_airquality.csv', parse_dates=['Date'], low_memory=False)
            actuals_df = source_df[source_df['COUNTY'] == 'Santa Clara'].groupby('Date')['DAILY_AQI_VALUE'].mean().reset_index()
            actuals_df.rename(columns={'DAILY_AQI_VALUE': 'Actual_AQI'}, inplace=True)

            # Merge the predictions and actuals on their common date
            comparison_df = pd.merge(actuals_df, predictions_df, on='Date', how='inner')
            comparison_df.set_index('Date', inplace=True)
            
            # UI slider to select how many days of history to show
            days_to_show = st.slider('Select number of days to display:', 1, 30, 14)
            
            # Display the comparison chart
            st.line_chart(comparison_df.tail(days_to_show))
            
            # Show the raw data in an expander
            with st.expander("View Raw Comparison Data"):
                st.dataframe(comparison_df.tail(days_to_show))

        except FileNotFoundError:
            st.warning("Prediction log not found. Run the data collection script for a few days to generate performance data.")
        except Exception as e:
            st.error(f"An error occurred while creating the performance chart: {e}")

    # --- AQI Legend Tab ---
    with tab_legend:
        st.header("Understanding the AQI Scale")
        st.markdown("""
            The U.S. EPA Air Quality Index (AQI) is a scale from 0 to 500. Higher values mean greater air pollution and health concern.
            
            - **0 - 50 (Good):** <span style='color:green; font-weight:bold;'>Green</span>
            - **51 - 100 (Moderate):** <span style='color:yellow; color:yellow; font-weight:bold;'>Yellow</span>
            - **101 - 150 (Unhealthy for Sensitive Groups):** <span style='color:orange; font-weight:bold;'>Orange</span>
            - **151 - 200 (Unhealthy):** <span style='color:red; font-weight:bold;'>Red</span>
            - **201 - 300 (Very Unhealthy):** <span style='color:purple; font-weight:bold;'>Purple</span>
            - **301+ (Hazardous):** <span style='color:maroon; font-weight:bold;'>Maroon</span>
        """, unsafe_allow_html=True)

    # --- About the Model Tab ---
    with tab_about:
        st.header("About the Prediction Model")
        st.info(f"The `{model_choice_name}` model was last retrained on: **{last_trained_date.strftime('%Y-%m-%d %H:%M:%S')}**")
        
        st.subheader("Model Type")
        st.write("We use a `RandomForestRegressor` model from the `scikit-learn` library, which is an ensemble of many decision trees.")

        st.subheader("Feature Engineering")
        st.write("The model doesn't just use yesterday's AQI. It predicts the future based on a set of engineered features, including:")
        st.markdown("""
            - **Lag Features:** The actual AQI values from each of the last 7 days.
            - **Rolling Window Features:** The mean and standard deviation of the AQI over the last 7 days.
            - **Putting it all together:**
                - "What was yesterday's AQI? And the day before? **(Lag features)**"
                - "What was the average AQI for the whole last week?" **(Rolling Mean)**
                - "Has the air quality been stable or chaotic recently?" **(Rolling Standard Deviation)**
        """)

        st.subheader("AQI Calculation Formula")
        st.write("To convert a raw pollutant concentration (like PM2.5) into the final 0-500 index, the US EPA uses a linear interpolation formula. This ensures that a concentration at the low end of a category gets a correspondingly low AQI score, and a concentration at the high end gets a high score.")
        st.latex(r'''
            I = \frac{I_{high} - I_{low}}{C_{high} - C_{low}}(C_p - C_{low}) + I_{low}
        ''')

        st.markdown("""
        Where:
        - **$I$** = The final, calculated Air Quality Index.
        - **$C_p$** = The measured pollutant concentration (in our case, PM2.5 in μg/m³).
        - **$C_{low}$** and **$C_{high}$** = The concentration breakpoints for the category that $C_p$ falls into (e.g., 12.1 and 35.4 for "Moderate").
        - **$I_{low}$** and **$I_{high}$** = The AQI value breakpoints corresponding to $C_{low}$ and $C_{high}$ (e.g., 51 and 100 for "Moderate").
        """)
        
        st.subheader("Training Data")
        st.write("The models were trained on the 'California Air Quality 2020' dataset from Kaggle, specifically for Santa Clara County. This dataset is notable for containing the extreme air quality events of the 2020 wildfire season.")

        st.subheader("The Two Models")
        st.markdown("""
            - **`model_2020.joblib`:** This was our initial model, trained only on "normal" air quality data (Jan-July 2020). It is naive and does not know how to handle extreme fire events.
            - **`model_fire_aware.joblib`:** This is the primary, more robust model. It was trained on the *entire* 2020 dataset, including the fire season. This gives it a "memory" of extreme events, making it more cautious and resilient. This model is re-trained once a week with the data of the previous week. 
        """)

# --- Main Entry Point ---
# This ensures the code below only runs when the script is executed directly.
if __name__ == "__main__":
    run_app()

