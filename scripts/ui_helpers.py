"""
Author: Lakshya Shrivastava

UI Helper Functions for the AQI Predictor Streamlit App.

This module contains utility functions specifically designed for enhancing the user
interface (UI) of the Streamlit application, such as functions that provide
dynamic styling based on data values
"""

def get_aqi_display_style(aqi):
    """
    Returns the background and text color CSS style for a given AQI value.

    Args:
        aqi (int or float): The AQI value.

    Returns:
        str: A CSS style string for use in HTML.
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