# Data Loader for Dashboard
import os
import pandas as pd

def smooth_data(data, window_size=100):
    """Smooth data using a simple moving average."""
    if data.empty:  # Check if the Series is empty
        return []

    smoothed = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return smoothed

