import pandas as pd

PATH_RAW_DATA_LOCATION = "data/raw/"

def get_raw_data():
    """
    Utility function to load the raw data from the data/raw folder.

    Returns:
        train_a (pd.DataFrame): The training targets for the A dataset.
        train_b (pd.DataFrame): The training targets for the B dataset.
        train_c (pd.DataFrame): The training targets for the C dataset.
        X_train_estimated_a (pd.DataFrame): The estimated training features for the A dataset.
        X_train_estimated_b (pd.DataFrame): The estimated training features for the B dataset.
        X_train_estimated_c (pd.DataFrame): The estimated training features for the C dataset.
        X_train_observed_a (pd.DataFrame): The observed training features for the A dataset.
        X_train_observed_b (pd.DataFrame): The observed training features for the B dataset.
        X_train_observed_c (pd.DataFrame): The observed training features for the C dataset.
        X_test_estimated_a (pd.DataFrame): The estimated test features for the A dataset.
        X_test_estimated_b (pd.DataFrame): The estimated test features for the B dataset.
        X_test_estimated_c (pd.DataFrame): The estimated test features for the C dataset.
    """
    train_a = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}A/train_targets.parquet')
    train_b = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}B/train_targets.parquet')
    train_c = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}C/train_targets.parquet')
    X_train_estimated_a = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}A/X_train_estimated.parquet')
    X_train_estimated_b = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}B/X_train_estimated.parquet')
    X_train_estimated_c = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}C/X_train_estimated.parquet')
    X_train_observed_a = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}A/X_train_observed.parquet')
    X_train_observed_b = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}B/X_train_observed.parquet')
    X_train_observed_c = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}C/X_train_observed.parquet')
    X_test_estimated_a = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet(f'{PATH_RAW_DATA_LOCATION}C/X_test_estimated.parquet')

    return train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c


def get_all_features() -> list:
    """
    Utility function to get all features from the raw data.

    Returns:
        features (list): A list of all features.
    """
    _, _, _, X_train_estimated_a, _, _, _, _, _, _, _, _ = get_raw_data()
    print(X_train_estimated_a.keys())
    all_features = X_train_estimated_a.keys()[0]
    all_featues = [feature for feature in all_features if feature != 'date_forecast']
    return all_featues

if __name__ == "__main__":
    data = get_raw_data()
    print(data[0].head())