import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


def get_naive_forecast(train_data, val_data, group_cols=("Store", "Dept")):
    """
    Naive Forecast (Last Known Value).
    For each (Store, Dept) group, forecasts the last observed sales value 
    from the training period for all validation periods.
    """
    train_df = train_data.copy()
    val_df = val_data.copy()
    
    # Ensure sorted by date
    train_df = train_df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    group_cols_list = list(group_cols)
    
    # Get the last record for each group
    last_known = train_df.groupby(group_cols_list).last().reset_index()
    last_known_map = last_known.set_index(group_cols_list)['Weekly_Sales'].to_dict()
    
    # Map predictions to validation dataframe
    val_df['Weekly_Sales_Pred'] = val_df.set_index(group_cols_list).index.map(last_known_map)
    
    # Fallback logic for groups not in training
    global_mean = train_df['Weekly_Sales'].mean()
    store_mean = train_df.groupby('Store')['Weekly_Sales'].mean().to_dict()
    
    # Fill missing predictions with store average or global average
    val_df['Weekly_Sales_Pred'] = val_df['Weekly_Sales_Pred'].fillna(val_df['Store'].map(store_mean))
    val_df['Weekly_Sales_Pred'] = val_df['Weekly_Sales_Pred'].fillna(global_mean)
    
    return val_df['Weekly_Sales_Pred'].values

def get_seasonal_naive_forecast(train_data, val_data, group_cols=("Store", "Dept")):
    """
    Seasonal Naive Forecast (Shift 52 Weeks).
    For each date in val_data, projects the sales of the same (Store, Dept) 
    from exactly 52 weeks (364 days) ago.
    """
    train_df = train_data.copy()
    val_df = val_data.copy()
    
    # Ensure Date columns are datetime objects
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    
    # Filter train_df to dates strictly before the minimum date in val_df to prevent leakage
    val_start_date = val_df['Date'].min()
    train_df = train_df[train_df['Date'] < val_start_date].reset_index(drop=True)
    
    # Compute the target date in the past (52 weeks ago = 364 days)
    val_df['Date_Shifted'] = val_df['Date'] - pd.Timedelta(days=364)
    
    # Extract only historical sales from train data
    group_cols_list = list(group_cols)
    history_df = train_df[train_df['Weekly_Sales'].notnull()][group_cols_list + ['Date', 'Weekly_Sales']]
    
    # Merge validation data with history based on Shifted Date
    merged = pd.merge(
        val_df,
        history_df,
        left_on=group_cols_list + ['Date_Shifted'],
        right_on=group_cols_list + ['Date'],
        how='left',
        suffixes=('', '_History')
    )
    
    # Rename historical sales as prediction
    predictions = merged['Weekly_Sales_History']
    
    # Fallbacks for missing historical records:
    # 1. Fall back to group's historical average sales
    group_mean = history_df.groupby(group_cols_list)['Weekly_Sales'].mean().reset_index()
    group_mean = group_mean.rename(columns={'Weekly_Sales': 'Weekly_Sales_Group_Mean'})
    merged = pd.merge(merged, group_mean, on=group_cols_list, how='left')
    predictions = predictions.fillna(merged['Weekly_Sales_Group_Mean'])
    
    # 2. Fall back to store-level average sales
    store_mean = history_df.groupby('Store')['Weekly_Sales'].mean().to_dict()
    predictions = predictions.fillna(merged['Store'].map(store_mean))
    
    # 3. Fall back to global average sales
    global_mean = history_df['Weekly_Sales'].mean()
    predictions = predictions.fillna(global_mean)
    
    return predictions.values


def _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict=None, global_mean=0.0):
    if len(group_train) == 0:
        if store_mean_dict is not None and len(val_df_group) > 0:
            store_id = val_df_group['Store'].iloc[0]
            fallback_val = store_mean_dict.get(store_id, global_mean)
        else:
            fallback_val = global_mean
        return np.full(len(val_df_group), fallback_val)
        
    group_train = group_train.copy()
    group_train['Date'] = pd.to_datetime(group_train['Date'])
    group_train = group_train.sort_values('Date')
    
    val_df_group = val_df_group.copy()
    val_df_group['Date'] = pd.to_datetime(val_df_group['Date'])
    
    # Fallback hierarchy:
    # 1. 52-week seasonal value (Seasonal Naive)
    val_df_group['Date_Shifted'] = val_df_group['Date'] - pd.Timedelta(days=364)
    history_map = group_train.set_index('Date')['Weekly_Sales'].to_dict()
    predictions = val_df_group['Date_Shifted'].map(history_map)
    
    # 2. Final training value (last known)
    last_val = group_train['Weekly_Sales'].iloc[-1]
    
    # 3. Group mean
    group_mean = group_train['Weekly_Sales'].mean()
    
    # 4. Store mean
    if store_mean_dict is not None and len(val_df_group) > 0:
        store_id = val_df_group['Store'].iloc[0]
        store_fallback = store_mean_dict.get(store_id, global_mean)
    else:
        store_fallback = global_mean
        
    # Combine fallbacks
    predictions = predictions.fillna(last_val).fillna(group_mean).fillna(store_fallback).fillna(global_mean)
    return predictions.values


def _prepare_group_series(group_train, val_dates):
    # Check for duplicate dates in the group
    if group_train["Date"].duplicated().any():
        raise ValueError("Duplicate dates found within Store-Dept group.")
        
    group_train = group_train.sort_values('Date')
    
    t_min = group_train['Date'].min()
    t_max = group_train['Date'].max()
    v_max = val_dates.max()
    
    # Reindex training series to a regular weekly grid
    full_train_idx = pd.date_range(start=t_min, end=t_max, freq='W-FRI')
    if len(full_train_idx) == 0:
        full_train_idx = pd.date_range(start=t_min, periods=len(group_train), freq='W-FRI')
        
    train_series = group_train.set_index('Date')['Weekly_Sales'].reindex(full_train_idx)
    
    # Calculate coverage ratio before filling missing values
    observed_ratio = train_series.notna().mean()
    
    # Fill missing values with 0.0 (no sales/closed department) as requested
    train_series = train_series.fillna(0.0).astype(float)
    
    # Determine forecast steps
    forecast_idx = pd.date_range(start=full_train_idx[-1] + pd.Timedelta(days=7), end=v_max, freq='W-FRI')
    steps = len(forecast_idx)
    if steps <= 0:
        steps = len(val_dates)
        forecast_idx = pd.date_range(start=full_train_idx[-1] + pd.Timedelta(days=7), periods=steps, freq='W-FRI')
        
    return train_series, steps, forecast_idx, observed_ratio


def _fit_predict_ses(group_train, val_df_group, store_mean_dict=None, global_mean=0.0, alpha=None):
    val_dates = pd.to_datetime(val_df_group['Date'])
    if len(group_train) < 2:
        return _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean), "cold_start", "Insufficient data (< 2 points)"
        
    try:
        train_series, steps, forecast_idx, observed_ratio = _prepare_group_series(group_train, val_dates)
        
        model = SimpleExpSmoothing(train_series, initialization_method="estimated")
        fit_kwargs = {}
        if alpha is not None:
            fit_kwargs['smoothing_level'] = alpha
        fit_model = model.fit(**fit_kwargs)
        pred = fit_model.forecast(steps)
        
        pred_series = pd.Series(pred, index=forecast_idx)
        val_pred = pred_series.reindex(val_dates).ffill().bfill().fillna(train_series.mean())
        return val_pred.clip(lower=0).values, "success", f"Observed ratio: {observed_ratio:.2%}"
    except Exception as e:
        fallback_val = _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean)
        return fallback_val, "model_failure", str(e)


def _fit_predict_holt(group_train, val_df_group, store_mean_dict=None, global_mean=0.0, smoothing_level=None, smoothing_trend=None):
    val_dates = pd.to_datetime(val_df_group['Date'])
    if len(group_train) < 2:
        return _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean), "cold_start", "Insufficient data (< 2 points)"
        
    try:
        train_series, steps, forecast_idx, observed_ratio = _prepare_group_series(group_train, val_dates)
        
        model = Holt(train_series, damped_trend=True, initialization_method="estimated")
        fit_kwargs = {}
        if smoothing_level is not None:
            fit_kwargs['smoothing_level'] = smoothing_level
        if smoothing_trend is not None:
            fit_kwargs['smoothing_trend'] = smoothing_trend
        fit_model = model.fit(optimized=True, **fit_kwargs)
        pred = fit_model.forecast(steps)
        
        pred_series = pd.Series(pred, index=forecast_idx)
        val_pred = pred_series.reindex(val_dates).ffill().bfill().fillna(train_series.mean())
        return val_pred.clip(lower=0).values, "success", f"Observed ratio: {observed_ratio:.2%}"
    except Exception as e:
        fallback_val = _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean)
        return fallback_val, "model_failure", str(e)


def _fit_predict_holt_winters(group_train, val_df_group, store_mean_dict=None, global_mean=0.0, seasonal_periods=52, trend="add", seasonal="add"):
    val_dates = pd.to_datetime(val_df_group['Date'])
    if len(group_train) < 2:
        return _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean), "cold_start", "Insufficient data (< 2 points)"
        
    try:
        train_series, steps, forecast_idx, observed_ratio = _prepare_group_series(group_train, val_dates)
        
        if len(train_series) < 2 * seasonal_periods:
            model = Holt(train_series, damped_trend=True, initialization_method="estimated")
            fit_model = model.fit(optimized=True)
            pred = fit_model.forecast(steps)
            status = "short_series"
            info = f"Length {len(train_series)} < 104. Fell back to damped Holt. Observed: {observed_ratio:.2%}"
        else:
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            )
            fit_model = model.fit()
            pred = fit_model.forecast(steps)
            status = "success"
            info = f"Observed ratio: {observed_ratio:.2%}"
            
        pred_series = pd.Series(pred, index=forecast_idx)
        val_pred = pred_series.reindex(val_dates).ffill().bfill().fillna(train_series.mean())
        return val_pred.clip(lower=0).values, status, info
    except Exception as e:
        fallback_val = _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean)
        return fallback_val, "model_failure", str(e)


def _fit_predict_arima(group_train, val_df_group, store_mean_dict=None, global_mean=0.0, order=(1, 1, 1)):
    val_dates = pd.to_datetime(val_df_group['Date'])
    if len(group_train) < 2:
        return _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean), "cold_start", "Insufficient data (< 2 points)"
        
    try:
        train_series, steps, forecast_idx, observed_ratio = _prepare_group_series(group_train, val_dates)
        
        model = ARIMA(train_series, order=order)
        fit_model = model.fit()
        pred = fit_model.forecast(steps)
        
        pred_series = pd.Series(pred, index=forecast_idx)
        val_pred = pred_series.reindex(val_dates).ffill().bfill().fillna(train_series.mean())
        return val_pred.clip(lower=0).values, "success", f"Observed ratio: {observed_ratio:.2%}"
    except Exception as e:
        fallback_val = _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean)
        return fallback_val, "model_failure", str(e)


def _fit_predict_sarima(group_train, val_df_group, store_mean_dict=None, global_mean=0.0, order=(1, 1, 0), seasonal_order=(0, 1, 1, 52)):
    val_dates = pd.to_datetime(val_df_group['Date'])
    if len(group_train) < 2:
        return _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean), "cold_start", "Insufficient data (< 2 points)"
        
    try:
        train_series, steps, forecast_idx, observed_ratio = _prepare_group_series(group_train, val_dates)
        
        seasonal_periods = seasonal_order[3]
        if len(train_series) < 2 * seasonal_periods:
            model = ARIMA(train_series, order=(1, 1, 1))
            fit_model = model.fit()
            pred = fit_model.forecast(steps)
            status = "short_series"
            info = f"Length {len(train_series)} < 104. Fell back to ARIMA(1,1,1). Observed: {observed_ratio:.2%}"
        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, simple_differencing=False)
            fit_model = model.fit(disp=False)
            pred = fit_model.forecast(steps)
            status = "success"
            info = f"Observed ratio: {observed_ratio:.2%}"
            
        pred_series = pd.Series(pred, index=forecast_idx)
        val_pred = pred_series.reindex(val_dates).ffill().bfill().fillna(train_series.mean())
        return val_pred.clip(lower=0).values, status, info
    except Exception as e:
        fallback_val = _get_group_fallback_forecast(group_train, val_df_group, store_mean_dict, global_mean)
        return fallback_val, "model_failure", str(e)


def _get_statistical_forecast_base(train_data, val_data, fit_predict_func, fit_predict_kwargs, group_cols=("Store", "Dept"), n_jobs=2):
    train_df = train_data.copy()
    val_df = val_data.copy()
    
    # Store original order of validation dataset
    val_df["_original_order"] = np.arange(len(val_df))
    
    # Ensure Date columns are datetime
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    
    # Sort
    train_df = train_df.sort_values(by=list(group_cols) + ['Date']).reset_index(drop=True)
    val_df = val_df.sort_values(by=list(group_cols) + ['Date']).reset_index(drop=True)
    
    # Pre-calculate global mean and store means for fallback use
    global_mean = train_df['Weekly_Sales'].mean()
    if pd.isna(global_mean):
        global_mean = 0.0
    store_mean_dict = train_df.groupby('Store')['Weekly_Sales'].mean().to_dict()
    
    # Identify unique groups in validation data
    val_groups = list(val_df.groupby(list(group_cols)))
    
    # Group training data for quick lookup
    train_grouped = {name: grp for name, grp in train_df.groupby(list(group_cols))}
    
    from joblib import Parallel, delayed
    
    def process_one_group(name, val_group):
        train_group = train_grouped.get(name)
        if train_group is None:
            fallback_val = _get_group_fallback_forecast(
                pd.DataFrame(columns=['Date', 'Weekly_Sales']),
                val_group,
                store_mean_dict,
                global_mean
            )
            return val_group.index, fallback_val, "cold_start", "No training data available"
            
        preds, status, info = fit_predict_func(
            train_group,
            val_group,
            store_mean_dict=store_mean_dict,
            global_mean=global_mean,
            **fit_predict_kwargs
        )
        return val_group.index, preds, status, info

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_group)(name, val_group)
        for name, val_group in val_groups
    )
    
    # Store predictions and collect fit statuses
    val_df['Weekly_Sales_Pred'] = 0.0
    status_counts = {"success": 0, "short_series": 0, "model_failure": 0, "cold_start": 0}
    
    for idxs, preds, status, info in results:
        val_df.loc[idxs, 'Weekly_Sales_Pred'] = preds
        status_counts[status] = status_counts.get(status, 0) + 1
        
    print(f"Model Fit Summary:")
    print(f"  Successful model fits:  {status_counts.get('success', 0)}")
    print(f"  Short-series fallbacks: {status_counts.get('short_series', 0)}")
    print(f"  Model failures:         {status_counts.get('model_failure', 0)}")
    print(f"  Cold-start groups:      {status_counts.get('cold_start', 0)}")
    
    # Restore original order before returning
    val_df = val_df.sort_values("_original_order")
    return val_df['Weekly_Sales_Pred'].to_numpy()


def get_ses_forecast(train_data, val_data, group_cols=("Store", "Dept"), alpha=None, n_jobs=-1):
    """
    Simple Exponential Smoothing forecast.
    """
    kwargs = {}
    if alpha is not None:
        kwargs['alpha'] = alpha
    return _get_statistical_forecast_base(train_data, val_data, _fit_predict_ses, kwargs, group_cols, n_jobs)


def get_holt_forecast(train_data, val_data, group_cols=("Store", "Dept"), smoothing_level=None, smoothing_trend=None, n_jobs=-1):
    """
    Holt's Linear Trend forecast.
    """
    kwargs = {}
    if smoothing_level is not None:
        kwargs['smoothing_level'] = smoothing_level
    if smoothing_trend is not None:
        kwargs['smoothing_trend'] = smoothing_trend
    return _get_statistical_forecast_base(train_data, val_data, _fit_predict_holt, kwargs, group_cols, n_jobs)


def get_holt_winters_forecast(train_data, val_data, group_cols=("Store", "Dept"), seasonal_periods=52, trend="add", seasonal="add", n_jobs=-1):
    """
    Holt-Winters (Exponential Smoothing with trend and seasonality) forecast.
    """
    kwargs = {
        'seasonal_periods': seasonal_periods,
        'trend': trend,
        'seasonal': seasonal
    }
    return _get_statistical_forecast_base(train_data, val_data, _fit_predict_holt_winters, kwargs, group_cols, n_jobs)


def get_arima_forecast(train_data, val_data, group_cols=("Store", "Dept"), order=(1, 1, 1), n_jobs=-1):
    """
    ARIMA forecast.
    """
    kwargs = {'order': order}
    return _get_statistical_forecast_base(train_data, val_data, _fit_predict_arima, kwargs, group_cols, n_jobs)


def get_sarima_forecast(train_data, val_data, group_cols=("Store", "Dept"), order=(1, 1, 0), seasonal_order=(0, 1, 1, 52), n_jobs=-1):
    """
    Seasonal ARIMA forecast.
    """
    kwargs = {
        'order': order,
        'seasonal_order': seasonal_order
    }
    return _get_statistical_forecast_base(train_data, val_data, _fit_predict_sarima, kwargs, group_cols, n_jobs)



