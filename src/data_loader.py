import pandas as pd
try:
    from src import config
except ModuleNotFoundError:
    import config

def load_raw_data():
    """Loads raw dataframes from the specified directory."""
    print("Loading raw data...")
    
    # Using low_memory=False to prevent warnings on large datasets
    train = pd.read_csv(config.TRAIN_RAW_PATH, low_memory=False)
    test = pd.read_csv(config.TEST_RAW_PATH, low_memory=False)
    stores = pd.read_csv(config.STORES_RAW_PATH, low_memory=False)
    features = pd.read_csv(config.FEATURES_RAW_PATH, low_memory=False)
    
    return train, test, stores, features
