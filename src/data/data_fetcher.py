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

def get_tests() -> pd.DataFrame:
    """
    Utility function to load the raw data from the data/raw folder.q
    """
    test = pd.read_csv(f'{PATH_RAW_DATA_LOCATION}test.csv')
    return test

def create_preprocessed_data():
    _, _, _, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, _, _, _ = get_raw_data()
    X_train_estimated_a = create_expected_pv_based_on_previous_years_same_day(X_train_estimated_a)
    X_train_estimated_b = create_expected_pv_based_on_previous_years_same_day(X_train_estimated_b)
    X_train_estimated_c = create_expected_pv_based_on_previous_years_same_day(X_train_estimated_c)
    X_train_observed_a = create_expected_pv_based_on_previous_years_same_day(X_train_observed_a)
    X_train_observed_b = create_expected_pv_based_on_previous_years_same_day(X_train_observed_b)
    X_train_observed_c = create_expected_pv_based_on_previous_years_same_day(X_train_observed_c)
    




def create_expected_pv_based_on_previous_years_same_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mean pv_measurement for each data point based on the previous years same day and hour
    Add this as a feature to the data frame
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least columns: 
                           location_a, location_b, location_c, date_forecast, 
                           pv_measurement, sin_day_of_year, cos_day_of_year, sin_hour, cos_hour
    
    Returns:
        pd.DataFrame: DataFrame with additional feature of mean pv_measurement based on historical data
    """
    df = df.copy()
    
    # Identify the location from the binary flags
    df['location'] = df[['location_a', 'location_b', 'location_c']].idxmax(axis=1)
    
    # Calculate mean pv_measurement for each location, sin_day_of_year, cos_day_of_year, sin_hour, and cos_hour
    mean_pv = df.groupby(['location', 'sin_day_of_year', 'cos_day_of_year', 'sin_hour', 'cos_hour'])['pv_measurement'].mean().reset_index()
    mean_pv.rename(columns={'pv_measurement': 'mean_pv_measurement'}, inplace=True)
    
    # Merge mean_pv_measurement back to the original DataFrame
    df = pd.merge(df, mean_pv, on=['location', 'sin_day_of_year', 'cos_day_of_year', 'sin_hour', 'cos_hour'], how='left')
    df.drop(columns=['location'], inplace=True)
    return df
if __name__ == "__main__":
    data = get_raw_data()
    print(data[0].head())