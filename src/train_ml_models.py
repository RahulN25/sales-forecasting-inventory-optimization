import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import lightgbm as lgb
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path

# Set seeds
np.random.seed(42)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.evaluate_models import evaluate_predictions
from src.train_models import (
    get_lr_forecast,
    get_rf_forecast,
    get_knn_forecast,
    get_xgb_forecast,
    get_lgb_forecast,
    get_mlp_forecast,
    get_ann_forecast,
    get_gbrt_forecast,
    get_ml_ensemble_forecast
)

def run_ml_pipeline():
    print("Starting ML Pipeline...")
    features_path = Path(config.TRAIN_FEATURES_PATH)
    df = pd.read_csv(features_path)
    
    # Standardize IsHoliday
    if df["IsHoliday"].dtype == "object":
        df["IsHoliday"] = (
            df["IsHoliday"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    df["IsHoliday"] = df["IsHoliday"].astype(bool)
    df["Date"] = pd.to_datetime(df["Date"])
    
    feature_cols = [
        'Store', 'Dept', 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price',
        'CPI', 'Unemployment', 'Year', 'Month', 'WeekOfYear', 'Quarter',
        'Is_Q4', 'Is_Holiday_Month', 'Is_SuperBowl', 'Is_LaborDay', 'Is_Thanksgiving', 'Is_Christmas',
        'MarkDown_Total', 'Has_MarkDown', 'MarkDown_Avg', 'MarkDown_Max',
        'Weekly_Sales_lag_1', 'Weekly_Sales_lag_2', 'Weekly_Sales_lag_3', 'Weekly_Sales_lag_4',
        'Weekly_Sales_lag_8', 'Weekly_Sales_lag_12', 'Weekly_Sales_lag_26', 'Weekly_Sales_lag_52',
        'rolling_mean_4', 'rolling_mean_12', 'rolling_mean_26', 'rolling_mean_52',
        'rolling_std_4', 'rolling_std_12', 'rolling_std_52',
        'Store_Avg_Sales', 'Dept_Avg_Sales', 'Store_Dept_Avg_Sales',
        'Type_Encoded', 'Holiday_MarkDown', 'Q4_Holiday', 'Size_MarkDown'
    ]
    
    df_clean = df.dropna(subset=['Weekly_Sales'] + feature_cols).copy()
    
    train_df = df_clean[df_clean["Date"] <= "2012-08-10"].copy()
    val_df = df_clean[df_clean["Date"] > "2012-08-10"].copy()
    y_val = val_df["Weekly_Sales"]
    
    metrics_list = []
    
    models = {
        "Linear Regression": lambda t, v, cols: get_lr_forecast(t, v, cols),
        "Random Forest": lambda t, v, cols: get_rf_forecast(t, v, cols),
        "KNN": lambda t, v, cols: get_knn_forecast(t, v, cols),
        "XGBoost": lambda t, v, cols: get_xgb_forecast(t, v, cols),
        "LightGBM": lambda t, v, cols: get_lgb_forecast(t, v, cols),
        "MLP": lambda t, v, cols: get_mlp_forecast(t, v, cols),
        "ANN": lambda t, v, cols: get_ann_forecast(t, v, cols),
        "GBRT": lambda t, v, cols: get_gbrt_forecast(t, v, cols),
        "Ensemble": lambda t, v, cols: get_ml_ensemble_forecast(t, v, cols)
    }
    
    for name, forecast_func in models.items():
        print(f"--- Running {name} ---")
        start_time = time.time()
        
        preds = forecast_func(train_df, val_df, feature_cols)
        runtime = time.time() - start_time
        
        col_name = f"{name}_Pred"
        val_df[col_name] = preds
        
        metrics = evaluate_predictions(
            y_true=y_val,
            y_pred=preds,
            is_holiday=val_df["IsHoliday"]
        )
        metrics["Model"] = name
        metrics["Runtime (s)"] = runtime
        metrics_list.append(metrics)
        
        print(f"  Completed in {runtime:.2f} seconds. WMAE: {metrics['WMAE']:,.2f}")
        
    # Save predictions
    output_dir = Path(config.BASE_DIR) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_path = output_dir / "ml_predictions.csv"
    val_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")
    
    # Save metrics
    metrics_path = output_dir / "ml_metrics.json"
    metrics_df = pd.DataFrame(metrics_list)[["Model", "WMAE", "MAE", "RMSE", "MAPE", "sMAPE", "Runtime (s)"]]
    metrics_df = metrics_df.sort_values("WMAE").reset_index(drop=True)
    metrics_dict = metrics_df.set_index("Model").to_dict(orient="index")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    run_ml_pipeline()
