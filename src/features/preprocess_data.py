import pandas as pd
from src.data.data_fetcher import get_raw_data
from src.features.feature_engineering import prepare_data, temporal_alignment


def fetch_preprocessed_data():
    train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()
    train_observed_a, train_estimated_a = temporal_alignment(train_a, X_train_observed_a, X_train_estimated_a)
    train_observed_b, train_estimated_b = temporal_alignment(train_b, X_train_observed_b, X_train_estimated_b)
    train_observed_c, train_estimated_c = temporal_alignment(train_c, X_train_observed_c, X_train_estimated_c)
    
    
    # Combine the temporally aligned datasets from all three locations
    train_observed_combined = pd.concat([train_observed_a, train_observed_b, train_observed_c], ignore_index=True)
    train_estimated_combined = pd.concat([train_estimated_a, train_estimated_b, train_estimated_c], ignore_index=True)
    
    # Prepare the combined dataset by handling missing values and splitting the data
    X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, \
    X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined = prepare_data(train_observed_combined, train_estimated_combined)
    return X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, \
    X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined

def get_preprocessed_test_data():
    
    train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()
    X_test_estimated_a_processed = X_test_estimated_a.drop(columns=['date_calc', 'date_forecast'])
    X_test_estimated_b_processed = X_test_estimated_b.drop(columns=['date_calc', 'date_forecast'])
    X_test_estimated_c_processed = X_test_estimated_c.drop(columns=['date_calc', 'date_forecast'])

    # Handle NaN values in the test data by filling them with the mean value of the respective column from the training data
    X_test_estimated_a_processed.fillna(0, inplace=True)
    X_test_estimated_b_processed.fillna(0, inplace=True)
    X_test_estimated_c_processed.fillna(0, inplace=True)
    return X_test_estimated_a_processed, X_test_estimated_b_processed, X_test_estimated_c_processed