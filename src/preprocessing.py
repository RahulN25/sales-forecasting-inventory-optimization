import pandas as pd
import numpy as np
from src import config
from src.data_loader import load_raw_data

def clean_data():
    """Performs the full data cleaning process for the Walmart dataset."""
    train, test, stores, features = load_raw_data()
    
    print("Parsing date columns...")
    # Convert dates to datetime objects
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])
    features['Date'] = pd.to_datetime(features['Date'])
    
    print("Handling missing values in features dataframe...")
    # 1. Markdown columns: filled with 0.0 (indicates no markdown/promotion active)
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in features.columns:
            # Replace string "NA" or "nan" with actual NaN, then fillna with 0.0
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
            
    # 2. CPI and Unemployment: 
    # Forward-fill (ffill) these values per store, then back-fill (bfill) if any remain at the start.
    features['CPI'] = pd.to_numeric(features['CPI'], errors='coerce')
    features['Unemployment'] = pd.to_numeric(features['Unemployment'], errors='coerce')
    
    # Sort features by Store and Date to ensure correct forward filling
    features = features.sort_values(by=['Store', 'Date']).reset_index(drop=True)
    features['CPI'] = features.groupby('Store')['CPI'].ffill()
    features['Unemployment'] = features.groupby('Store')['Unemployment'].ffill()
    features['CPI'] = features.groupby('Store')['CPI'].bfill()
    features['Unemployment'] = features.groupby('Store')['Unemployment'].bfill()

    print("Merging dataframes...")
    # Convert IsHoliday columns to boolean to ensure clean merges
    train['IsHoliday'] = train['IsHoliday'].astype(bool)
    test['IsHoliday'] = test['IsHoliday'].astype(bool)
    features['IsHoliday'] = features['IsHoliday'].astype(bool)
    
    # Merge train and test with stores (adds Type and Size)
    train_merged = pd.merge(train, stores, on='Store', how='left')
    test_merged = pd.merge(test, stores, on='Store', how='left')
    
    # Merge with features on Store, Date, and IsHoliday.
    train_cleaned = pd.merge(train_merged, features, on=['Store', 'Date', 'IsHoliday'], how='left')
    test_cleaned = pd.merge(test_merged, features, on=['Store', 'Date', 'IsHoliday'], how='left')
    
    # Save the cleaned datasets
    print(f"Saving cleaned train data to {config.TRAIN_CLEANED_PATH}...")
    train_cleaned.to_csv(config.TRAIN_CLEANED_PATH, index=False)
    
    print(f"Saving cleaned test data to {config.TEST_CLEANED_PATH}...")
    test_cleaned.to_csv(config.TEST_CLEANED_PATH, index=False)
    
    print("Data cleaning complete!")
    return train_cleaned, test_cleaned

if __name__ == "__main__":
    clean_data()
