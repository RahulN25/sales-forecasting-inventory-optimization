import pandas as pd
import numpy as np
import sys
import os

# Append project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_models import (
    get_ses_forecast,
    get_holt_forecast,
    get_holt_winters_forecast,
    get_arima_forecast,
    get_sarima_forecast
)

def generate_synthetic_data():
    print("Generating synthetic dataset...")
    # Generate 120 weeks of Friday dates
    dates = pd.date_range(start="2010-01-01", periods=120, freq="W-FRI")
    
    # Store 1, Dept 1
    # Sales with a trend and annual seasonality (52 weeks)
    sales_1 = [10000 + 100 * i + 2000 * np.sin(2 * np.pi * i / 52) + np.random.normal(0, 500) for i in range(120)]
    
    # Store 1, Dept 2
    # Sales with trend, no seasonality
    sales_2 = [20000 - 50 * i + np.random.normal(0, 1000) for i in range(120)]
    
    data_list = []
    for d, s in zip(dates, sales_1):
        data_list.append({"Store": 1, "Dept": 1, "Date": d, "Weekly_Sales": max(0, s)})
    for d, s in zip(dates, sales_2):
        data_list.append({"Store": 1, "Dept": 2, "Date": d, "Weekly_Sales": max(0, s)})
        
    df = pd.DataFrame(data_list)
    
    # Train/Val Split (last 12 weeks for validation)
    train_df = df.iloc[:-24]  # 96 weeks
    val_df = df.iloc[-24:]    # 12 weeks for each group (24 rows total)
    
    return train_df, val_df

def test_models():
    train_df, val_df = generate_synthetic_data()
    
    models = {
        "Simple Exponential Smoothing": lambda t, v: get_ses_forecast(t, v),
        "Holt's Linear Trend": lambda t, v: get_holt_forecast(t, v),
        "Holt-Winters": lambda t, v: get_holt_winters_forecast(t, v, seasonal_periods=52),
        "ARIMA": lambda t, v: get_arima_forecast(t, v),
        "SARIMA": lambda t, v: get_sarima_forecast(t, v, seasonal_order=(1, 0, 0, 52))
    }
    
    print("\nRunning model verification tests...")
    
    for name, forecast_func in models.items():
        print(f"Testing {name}...")
        try:
            preds = forecast_func(train_df, val_df)
            
            # Assertions
            assert len(preds) == len(val_df), f"Expected forecast length {len(val_df)}, but got {len(preds)}"
            assert not np.isnan(preds).any(), f"Forecast contains NaN values"
            assert not np.isinf(preds).any(), f"Forecast contains Inf values"
            assert (preds >= 0).all(), f"Forecast contains negative sales predictions"
            
            print(f"  [SUCCESS] {name} forecast check passed. Prediction sample: {preds[:3]}")
        except Exception as e:
            print(f"  [FAILURE] {name} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print("\nAll statistical models successfully verified!")

if __name__ == "__main__":
    test_models()
