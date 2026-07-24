import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import lightgbm
import numpy as np
import pandas as pd

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    Filters out actual values that are zero to avoid division by zero.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_smape(y_true, y_pred, epsilon=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > epsilon
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

def calculate_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (WAPE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0
    return (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100

def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Weighted Mean Absolute Error (WMAE) used in the Walmart Kaggle Competition.
    Holiday weeks have a weight of 5, non-holiday weeks have a weight of 1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Map boolean or numeric holiday indicator to weights (5 for holiday, 1 for regular)
    is_holiday = np.array(is_holiday)
    weights = np.where(is_holiday == 1, 5, 1)
    
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def evaluate_predictions(y_true, y_pred, is_holiday=None):
    """
    Evaluates predictions using all defined metrics.
    Returns a dictionary of metrics.
    """
    metrics = {
        "MAE": float(calculate_mae(y_true, y_pred)),
        "RMSE": float(calculate_rmse(y_true, y_pred)),
        "MAPE": float(calculate_mape(y_true, y_pred)),
        "sMAPE": float(calculate_smape(y_true, y_pred)),
        "WAPE": float(calculate_wape(y_true, y_pred)),
    }
    
    if is_holiday is not None:
        metrics["WMAE"] = float(calculate_wmae(y_true, y_pred, is_holiday))
        
    return metrics

def get_train_val_split(df, val_start_date="2012-08-10", val_end_date="2012-10-26"):
    """
    Splits the dataframe into training and validation sets based on dates.
    Assumes df has a 'Date' column which will be converted to datetime.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    
    val_start = pd.to_datetime(val_start_date)
    val_end = pd.to_datetime(val_end_date)
    
    train_df = df[df["Date"] < val_start].reset_index(drop=True)
    val_df = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].reset_index(drop=True)
    
    return train_df, val_df
