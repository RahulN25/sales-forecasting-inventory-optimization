import pandas as pd
import numpy as np

def create_date_features(df):
    """
    Extracts date-related components from the 'Date' column.
    Assumes df['Date'] is already converted to datetime.
    """
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['Is_Q4'] = (df['Quarter'] == 4).astype(int)
    # Holiday months are Feb (Super Bowl), Sep (Labor Day), Nov (Thanksgiving), Dec (Christmas)
    df['Is_Holiday_Month'] = df['Month'].isin([2, 9, 11, 12]).astype(int)
    return df


def create_holiday_details(df):
    """
    Creates specific binary indicators for major Walmart holidays.
    Dates are mapped from historical Walmart holiday periods:
    - Super Bowl: Feb 12, 2010; Feb 11, 2011; Feb 10, 2012; Feb 8, 2013
    - Labor Day: Sep 10, 2010; Sep 9, 2011; Sep 7, 2012; Sep 6, 2013
    - Thanksgiving: Nov 26, 2010; Nov 25, 2011; Nov 23, 2012; Nov 29, 2013
    - Christmas: Dec 31, 2010; Dec 30, 2011; Dec 28, 2012; Dec 27, 2013
    """
    df = df.copy()
    
    super_bowl_dates = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'])
    labor_day_dates = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'])
    thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'])
    christmas_dates = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'])
    
    df['Is_SuperBowl'] = df['Date'].isin(super_bowl_dates).astype(int)
    df['Is_LaborDay'] = df['Date'].isin(labor_day_dates).astype(int)
    df['Is_Thanksgiving'] = df['Date'].isin(thanksgiving_dates).astype(int)
    df['Is_Christmas'] = df['Date'].isin(christmas_dates).astype(int)
    
    return df


def create_lag_features(df, lags=[1, 2, 3, 4, 8, 12, 26, 52]):
    """
    Computes weekly sales lag features.
    Lags are computed grouped by Store and Dept to keep time series separate.
    """
    df = df.copy()
    df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    for lag in lags:
        df[f'Weekly_Sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
        
    return df


def create_rolling_features(df):
    """
    Computes rolling mean and standard deviation on 1-week shifted sales.
    Rolling calculations are grouped by Store and Dept to avoid data leakage.
    """
    df = df.copy()
    df = df.sort_values(by=["Store", "Dept", "Date"]).reset_index(drop=True)

    group_cols = ["Store", "Dept"]

    for w in [4, 12, 26, 52]:
        df[f"rolling_mean_{w}"] = (
            df.groupby(group_cols)["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )

    for w in [4, 12, 52]:
        df[f"rolling_std_{w}"] = (
            df.groupby(group_cols)["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=2).std())
            .fillna(0.0)
        )

    return df


def create_markdown_features(df):
    """
    Aggregates markdown statistics.
    Computes total sum, has_markdown flag, average, and maximum of active markdowns.
    """
    df = df.copy()
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    md_present = [col for col in markdown_cols if col in df.columns]
    
    if md_present:
        df['MarkDown_Total'] = df[md_present].sum(axis=1)
        df['Has_MarkDown'] = (df['MarkDown_Total'] > 0).astype(int)
        df['MarkDown_Avg'] = df[md_present].mean(axis=1)
        df['MarkDown_Max'] = df[md_present].max(axis=1)
    else:
        df['MarkDown_Total'] = 0.0
        df['Has_MarkDown'] = 0
        df['MarkDown_Avg'] = 0.0
        df['MarkDown_Max'] = 0.0
        
    return df


def create_historical_averages(df, train_part):
    """
    Computes store-level, department-level, and store-department-level historical average sales.
    Computed using only train_part to avoid leakage from the test target.
    """
    df = df.copy()
    
    # Store average sales
    store_avg = train_part.groupby('Store')['Weekly_Sales'].mean().to_dict()
    df['Store_Avg_Sales'] = df['Store'].map(store_avg)
    
    # Department average sales
    dept_avg = train_part.groupby('Dept')['Weekly_Sales'].mean().to_dict()
    df['Dept_Avg_Sales'] = df['Dept'].map(dept_avg)
    
    # Store-Department average sales
    store_dept_avg = train_part.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
    store_dept_avg = store_dept_avg.rename(columns={'Weekly_Sales': 'Store_Dept_Avg_Sales'})
    df = df.merge(store_dept_avg, on=['Store', 'Dept'], how='left')
    
    # Fallbacks in case test has unseen stores/departments
    df['Store_Avg_Sales'] = df['Store_Avg_Sales'].fillna(train_part['Weekly_Sales'].mean())
    df['Dept_Avg_Sales'] = df['Dept_Avg_Sales'].fillna(train_part['Weekly_Sales'].mean())
    df['Store_Dept_Avg_Sales'] = df['Store_Dept_Avg_Sales'].fillna(df['Store_Avg_Sales'])
    
    return df


def create_interaction_features(df):
    """
    Constructs interaction features between categorical, numerical, and economic flags.
    """
    df = df.copy()

    # Safely convert IsHoliday to integer flag
    if df["IsHoliday"].dtype == bool:
        holiday_flag = df["IsHoliday"].astype(int)
    else:
        holiday_flag = (
            df["IsHoliday"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

    df["Holiday_MarkDown"] = holiday_flag * df["Has_MarkDown"]
    df["Q4_Holiday"] = df["Is_Q4"] * holiday_flag
    df["Size_MarkDown"] = df["Size"] * df["MarkDown_Total"]

    return df


def build_features_pipeline(df):
    """
    Consolidated function to apply all feature engineering steps in sequence.
    """
    df = df.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
    
    # Identify training records to calculate historical averages without target leakage
    train_part = df[df['Weekly_Sales'].notnull()]
    
    df = create_date_features(df)
    df = create_holiday_details(df)
    df = create_markdown_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_historical_averages(df, train_part)
    
    # Store Type categorical label encoding
    if 'Type' in df.columns:
        df['Type_Encoded'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3})
        
    df = create_interaction_features(df)
        
    return df
