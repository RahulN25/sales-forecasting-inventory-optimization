import pandas as pd
from pathlib import Path

try:
    from src import config
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    import config
    from data_loader import load_raw_data


def clean_holiday_column(series):
    """Safely convert IsHoliday column to boolean."""
    if series.dtype == bool:
        return series

    return series.astype(str).str.strip().str.lower().map({
        "true": True,
        "false": False,
        "1": True,
        "0": False
    })


def clean_data():
    """Performs the full data cleaning process for the Walmart dataset."""

    train, test, stores, features = load_raw_data()

    print("Parsing date columns...")
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    print("Converting numeric columns...")

    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    numeric_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"] + markdown_cols

    for col in numeric_cols:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    print("Handling missing markdown values...")
    for col in markdown_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)

    print("Handling missing CPI and Unemployment values...")
    features = features.sort_values(by=["Store", "Date"]).reset_index(drop=True)

    for col in ["CPI", "Unemployment"]:
        features[col] = features.groupby("Store")[col].ffill()
        features[col] = features.groupby("Store")[col].bfill()

    print("Cleaning IsHoliday column...")
    train["IsHoliday"] = clean_holiday_column(train["IsHoliday"])
    test["IsHoliday"] = clean_holiday_column(test["IsHoliday"])
    features["IsHoliday"] = clean_holiday_column(features["IsHoliday"])

    print("Merging datasets...")

    train_merged = pd.merge(train, stores, on="Store", how="left")
    test_merged = pd.merge(test, stores, on="Store", how="left")

    # Merge on Store and Date only to avoid mismatch issues with IsHoliday
    feature_cols = [col for col in features.columns if col != "IsHoliday"]

    train_cleaned = pd.merge(
        train_merged,
        features[feature_cols],
        on=["Store", "Date"],
        how="left"
    )

    test_cleaned = pd.merge(
        test_merged,
        features[feature_cols],
        on=["Store", "Date"],
        how="left"
    )

    print("Checking merged data...")
    assert len(train_cleaned) == len(train), "Train row count changed after merge!"
    assert len(test_cleaned) == len(test), "Test row count changed after merge!"

    print("Missing values after cleaning:")
    print(train_cleaned.isnull().sum())

    print("Saving cleaned datasets...")
    Path(config.TRAIN_CLEANED_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.TEST_CLEANED_PATH).parent.mkdir(parents=True, exist_ok=True)

    train_cleaned.to_csv(config.TRAIN_CLEANED_PATH, index=False)
    test_cleaned.to_csv(config.TEST_CLEANED_PATH, index=False)

    print("Data cleaning complete!")

    return train_cleaned, test_cleaned


if __name__ == "__main__":
    clean_data()