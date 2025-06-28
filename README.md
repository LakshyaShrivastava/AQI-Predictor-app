# 💨AQI Predictor for Santa Clara County

Link to app: https://ls-aqi-predictor.streamlit.app/

##Project Overview
This project is an end-to-end machine learning application designed to predict the daily Air Quality Index (AQI) for Santa Clara County, California. It features an adaptive MLOps pipeline that automatically collects new data daily and retrains the model weekly, allowing it to improve and adapt over time.

The project was built to explore time-series forecasting, handle real-world data issues, and implement a complete, automated ML system from data collection to deployment.

##✨ Features

**7-Day AQI Forecasting:** Predicts the AQI for the next seven days using a recursive forecasting strategy.

**Live Data Integration:** Uses the OpenWeatherMap API to fetch real, recent air quality data to use as a baseline for predictions.

**Robust Model Comparison:** Allows users to switch between two models to see the impact of training data on predictions:

- **Standard Model: **Trained only on "normal" air quality data.
- **Fire-Aware Model:** Trained on data that includes the extreme 2020 California wildfire season, making it more robust. This is also updated every week with current data.

**Automated MLOps Pipeline:** Powered by GitHub Actions, the system automatically
- **Collects data daily: **Appends yesterday's verified AQI to the dataset.
- **Retrains the model weekly:** Creates a new, smarter model using all the latest data.

**Interactive Web UI:** A user-friendly dashboard built with Streamlit to visualize the forecast.

##🛠️ How It Works
The system follows a continuous loop, making it an adaptive application:

**Daily API Call -> Append to CSV Dataset -> Weekly Model Retraining -> Save New Model File -> Streamlit App Uses Latest Model**
