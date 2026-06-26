import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw" / "wallmart-store-sales-forecasting"
DATA_INTERIM_DIR = BASE_DIR / "data" / "interim"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure directories exist
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_RAW_PATH = DATA_RAW_DIR / "train.csv"
TEST_RAW_PATH = DATA_RAW_DIR / "test.csv"
STORES_RAW_PATH = DATA_RAW_DIR / "stores.csv"
FEATURES_RAW_PATH = DATA_RAW_DIR / "features.csv"

# Output Files
TRAIN_CLEANED_PATH = DATA_PROCESSED_DIR / "train_cleaned.csv"
TEST_CLEANED_PATH = DATA_PROCESSED_DIR / "test_cleaned.csv"
