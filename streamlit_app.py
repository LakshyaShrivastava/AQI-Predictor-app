import streamlit as st
import pandas as pd
import joblib
from collections import deque
from datetime import datetime, timedelta
import os

# --- Import our custom helper functions ---
from scripts.predict_helpers import get_historical_pm25, pm25_to_aqi, create_features_for_prediction

# --- Page Configuration (Set this at the very top) ---
st.set_page_config(
    page_title="AQI Forecaster",
    page_icon="💨",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def fetch_recent_data(lat, lon, api_key, n_days):
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

# --- Sidebar for Settings ---
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a Prediction Model:",
    ('model_santa_clara_fire_aware.joblib', 'model_2020.joblib')
)

model_choice = "./models/" + model_choice 

def get_aqi_display_style(aqi):
    """
    Returns the background and text color style for a given AQI value based on EPA standards.
    """
    aqi = int(aqi) # Ensure AQI is an integer
    if aqi <= 50:
        # Good - Green
        return "background-color: #2b9348; color: white;"
    elif aqi <= 100:
        # Moderate - Yellow
        return "background-color: #f7d44c; color: yellow;"
    elif aqi <= 150:
        # Unhealthy for Sensitive Groups - Orange
        return "background-color: #f89938; color: black;"
    elif aqi <= 200:
        # Unhealthy - Red
        return "background-color: #f26666; color: white;"
    elif aqi <= 300:
        # Very Unhealthy - Purple
        return "background-color: #a37ac4; color: white;"
    else:
        # Hazardous - Maroon
        return "background-color: #931f1f; color: white;"

# --- Main Application Logic ---
# This part is wrapped in a main function for clarity
def run_app():
    # Set up constants and API Key
    TARGET_COLUMN = 'DAILY_AQI_VALUE'
    N_LAGS = 7
    API_KEY = st.secrets.get("OWM_API_KEY")

    if not API_KEY:
        st.error("OWM_API_KEY not found. Please add it to your Streamlit Secrets.")
        st.stop()

    # Load model and display last trained time
    model = load_model(model_choice)
    if model is None:
        st.error(f"Model file '{model_choice}' not found.")
        st.stop()
    
    model_path = model_choice
    last_trained_timestamp = os.path.getmtime(model_path)
    last_trained_date = datetime.fromtimestamp(last_trained_timestamp)

    # Fetch data and perform forecast
    last_7_days_aqi = fetch_recent_data(37.4323, -121.8996, API_KEY, N_LAGS)
    current_window = deque(last_7_days_aqi, maxlen=N_LAGS)
    future_predictions = []
    for _ in range(7):
        input_features = create_features_for_prediction(current_window, TARGET_COLUMN, N_LAGS)
        predicted_aqi = model.predict(input_features)[0]
        future_predictions.append(predicted_aqi)
        current_window.append(predicted_aqi)

    # Create a DataFrame for the forecast
    start_date = datetime.now() + timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, periods=7)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted AQI': future_predictions}).set_index('Date')

    # --- TABS FOR LAYOUT ---
    tab_forecast, tab_legend, tab_about = st.tabs(["📈 Forecast", "🎨 AQI Legend", "🤖 About the Model"])

    with tab_forecast:
        st.header(f"7-Day Forecast using `{model_choice}`")
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

    with tab_about:
        st.header("About the Prediction Model")
        st.info(f"The `{model_choice}` model was last retrained on: **{last_trained_date.strftime('%Y-%m-%d %H:%M:%S')}**")
        
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
        
        st.subheader("Training Data")
        st.write("The models were trained on the 'California Air Quality 2020' dataset from Kaggle, specifically for Santa Clara County. This dataset is notable for containing the extreme air quality events of the 2020 wildfire season.")

        st.subheader("The Two Models")
        st.markdown("""
            - **`model_2020.joblib`:** This was our initial model, trained only on "normal" air quality data (Jan-July 2020). It is naive and does not know how to handle extreme fire events.
            - **`model_fire_aware.joblib`:** This is the primary, more robust model. It was trained on the *entire* 2020 dataset, including the fire season. This gives it a "memory" of extreme events, making it more cautious and resilient. This model is re-trained once a week with the data of the previous week. 
        """)

# --- Run the main function ---
if __name__ == "__main__":
    st.title("💨 Santa Clara County AQI Forecaster")
    st.markdown("An adaptive machine learning model for daily Air Quality Index (AQI) prediction.")
    run_app()

