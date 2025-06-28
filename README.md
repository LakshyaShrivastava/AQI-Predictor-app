# 💨AQI Predictor for Santa Clara County

Link to app: https://ls-aqi-predictor.streamlit.app/

## Project Overview
This project is an end-to-end machine learning application designed to predict the daily Air Quality Index (AQI) for Santa Clara County, California. It features an adaptive MLOps pipeline that automatically collects new data daily and retrains the model weekly, allowing it to improve and adapt over time.

The project was built to explore time-series forecasting, handle real-world data issues, and implement a complete, automated ML system from data collection to deployment.

## ✨ Features

**7-Day AQI Forecasting:** Predicts the AQI for the next seven days using a recursive forecasting strategy.

**Live Data Integration:** Uses the OpenWeatherMap API to fetch real, recent air quality data to use as a baseline for predictions.

**Robust Model Comparison:** Allows users to switch between two models to see the impact of training data on predictions:

- **Standard Model:** Trained only on "normal" air quality data.
- **Fire-Aware Model:** Trained on data that includes the extreme 2020 California wildfire season, making it more robust. This is also updated every week with current data.

**Automated MLOps Pipeline:** Powered by GitHub Actions, the system automatically
- **Collects data daily:** Appends yesterday's verified AQI to the dataset.
- **Retrains the model weekly:** Creates a new, smarter model using all the latest data.

**Interactive Web UI:** A user-friendly dashboard built with Streamlit to visualize the forecast.

## 🛠️ How It Works
The system follows a continuous loop, making it an adaptive application:

**Daily API Call -> Append to CSV Dataset -> Weekly Model Retraining -> Save New Model File -> Streamlit App Uses Latest Model**

## 🚀 Getting Started: Running the Project Locally
To set up and run this application on your local machine, please follow these instructions carefully.

**Prerequisites**

Before you begin, ensure you have the following software installed on your system:

- **Python:** Version 3.9 or higher is recommended.

- **Git:** For cloning the repository from GitHub.

**Step-by-Step Instructions**
**1. Clone the Repository**

``` Bash
git clone https://github.com/LakshyaShrivastava/AQI-Predictor-app.git
cd AQI-Predictor-app
```

**2. Create and Activate a Virtual Environment**

It is a best practice to create an isolated Python environment for the project's dependencies.

```Bash
# Create the virtual environment folder named 'venv'
python -m venv venv
```
Now, activate the environment. The command differs based on your operating system:

**On Windows (PowerShell or Command Prompt):**

```PowerShell
.\venv\Scripts\activate
```
**On macOS / Linux:**

```Bash
source venv/bin/activate
```
Your terminal prompt should now be prefixed with (venv).

**3. Install Required Packages**

This project uses a `requirements.txt` file to manage all its dependencies. Run the following command to install them all at once:

```Bash
pip install -r requirements.txt
```

**4. Set Up Your Local API Key**

The application requires an API key from OpenWeatherMap to fetch live air quality data. The app is designed to read this key from a local secrets file.

- In the main project directory, create a new folder named .streamlit (note the dot at the beginning).

- Inside the .streamlit folder, create a new file named secrets.toml.

- Add your API key to this secrets.toml file in the following format:

	```
	OWM_API_KEY = "your_actual_api_key_goes_here"
	```

Replace the placeholder with your actual key.

**5.Run the Streamlit Application**

You're all set! Make sure your virtual environment is still active and run the following command to launch the app:

```Bash
streamlit run app.py
```
Your default web browser should automatically open a new tab with the AQI Predictor application running. You can now interact with the UI, select different models, and see the forecast.
