💨 AQI Predictor for Santa Clara County

(Note: Replace the URL above with your actual deployed Streamlit app URL)

Project Overview
This project is an end-to-end machine learning application designed to predict the daily Air Quality Index (AQI) for Santa Clara County, California. It features an adaptive MLOps pipeline that automatically collects new data daily and retrains the model weekly, allowing it to improve and adapt over time.

The project was built to explore time-series forecasting, handle real-world data issues, and implement a complete, automated ML system from data collection to deployment.

✨ Features
7-Day AQI Forecasting: Predicts the AQI for the next seven days using a recursive forecasting strategy.

Live Data Integration: Uses the OpenWeatherMap API to fetch real, recent air quality data to use as a baseline for predictions.

Robust Model Comparison: Allows users to switch between two models to see the impact of training data on predictions:

Standard Model: Trained only on "normal" air quality data.

Fire-Aware Model: Trained on data that includes the extreme 2020 California wildfire season, making it more robust. This is also updated every week with current data.

Automated MLOps Pipeline: Powered by GitHub Actions, the system automatically:

Collects data daily: Appends yesterday's verified AQI to the dataset.

Retrains the model weekly: Creates a new, smarter model using all the latest data.

Interactive Web UI: A user-friendly dashboard built with Streamlit to visualize the forecast.

🛠️ How It Works
The system follows a continuous loop, making it an adaptive application:

Daily API Call -> Append to CSV Dataset -> Weekly Model Retraining -> Save New Model File -> Streamlit App Uses Latest Model

🚀 Getting Started Locally
To run this project on your own machine, follow these steps:

1. Prerequisites

Python 3.8+

Git

2. Clone the Repository

Bash

git clone https://github.com/LakshyaShrivastava/AQI-Predictor-app.git
cd AQI-Predictor-app
3. Set up the Virtual Environment

Bash

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Or (macOS/Linux)
# source venv/bin/activate
4. Install Dependencies

Bash

pip install -r requirements.txt
5. Set Up Local Secrets

Create a folder named .streamlit in the main project directory.

Inside that folder, create a file named secrets.toml.

Add your OpenWeatherMap API key to this file:

Ini, TOML

OWM_API_KEY = "your_actual_api_key_goes_here"
Important: The .gitignore file is already configured to prevent this secrets file from being committed to GitHub.

6. Run the Streamlit App

Bash

streamlit run app.py
Your web browser will open with the local version of the application.

📁 Project Structure
.
├── .github/
│   └── workflows/
│       ├── collect_data.yml    # GitHub Action for daily data collection
│       └── retrain_model.yml   # GitHub Action for weekly model retraining
├── .streamlit/
│   └── secrets.toml          # Local secrets file (not committed)
├── app.py                    # The main Streamlit web application
├── collect_data.py           # Script to fetch and append new daily data
├── main.py                     # Script to train the ML model
├── predict_helpers.py        # Utility functions for API calls, AQI conversion, etc.
├── model_fire_aware.joblib   # The primary, robust prediction model
├── requirements.txt          # Project dependencies
└── README.md                 # This file
🤖 Automation Pipeline
This project uses a GitHub Actions-powered MLOps pipeline to stay up-to-date.

collect_data.yml: This workflow runs daily. It executes collect_data.py to fetch the previous day's AQI, converts it to the US EPA standard, and appends it to the master CSV dataset, committing the change back to the repository.

retrain_model.yml: This workflow runs weekly (every Thursday at 9 PM PDT). It executes main.py to train a new Random Forest model on the entire updated dataset and commits the new model_fire_aware.joblib file, replacing the old one.

When the model file is updated on the main branch, the Streamlit Community Cloud app automatically re-deploys, ensuring the live application is always using the latest and smartest model.
