# Data Loader for Dashboard
import os
import pandas as pd

def smooth_data(data, window_size=100):
    """Smooth data using a simple moving average."""
    if data.empty:  # Check if the Series is empty
        return []

    smoothed = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return smoothed

def get_directories(path):
    """Get a list of sub-directories in a given directory."""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def load_data(filepath):
    return pd.read_csv(filepath)