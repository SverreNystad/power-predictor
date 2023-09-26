import pandas as pd
from src.data.data_fetcher import get_raw_data, get_tests
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

def get_final_prediction(predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, predictions_c: pd.DataFrame) -> pd.DataFrame:
    """
    Get the final prediction by combining the predictions for the three locations.
    """
    tests = get_tests()
    X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_preprocessed_test_data_with_time()

    # Ensure 'time' columns have the same data type before merging
    tests['time'] = pd.to_datetime(tests['time'])
    X_test_estimated_a['date_forecast'] = pd.to_datetime(X_test_estimated_a['date_forecast'])
    X_test_estimated_b['date_forecast'] = pd.to_datetime(X_test_estimated_b['date_forecast'])
    X_test_estimated_c['date_forecast'] = pd.to_datetime(X_test_estimated_c['date_forecast'])
    
    # Drop the 'prediction' column from 'tests' DataFrame if it exists
    tests = tests.drop(columns='prediction', errors='ignore')
    
    # Create DataFrames for each set of predictions with 'time', 'location', and 'prediction' columns
    df_pred_a = pd.DataFrame({'time': X_test_estimated_a['date_forecast'], 'location': 'A', 'prediction': predictions_a})
    df_pred_b = pd.DataFrame({'time': X_test_estimated_b['date_forecast'], 'location': 'B', 'prediction': predictions_b})
    df_pred_c = pd.DataFrame({'time': X_test_estimated_c['date_forecast'], 'location': 'C', 'prediction': predictions_c})

    # Concatenate the prediction DataFrames vertically
    df_all_predictions = pd.concat([df_pred_a, df_pred_b, df_pred_c], ignore_index=True)

    # Merge with the 'test' DataFrame on 'time' and 'location' to fill in the 'prediction' column in the 'test' DataFrame
    test = tests.merge(df_all_predictions, on=['time', 'location'], how='left')
    return test


def get_preprocessed_test_data_with_time():
    train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()
    X_test_estimated_a_processed = X_test_estimated_a.drop(columns=['date_calc'])
    X_test_estimated_b_processed = X_test_estimated_b.drop(columns=['date_calc'])
    X_test_estimated_c_processed = X_test_estimated_c.drop(columns=['date_calc'])

    # Handle NaN values in the test data by filling them with the mean value of the respective column from the training data
    X_test_estimated_a_processed.fillna(0, inplace=True)
    X_test_estimated_b_processed.fillna(0, inplace=True)
    X_test_estimated_c_processed.fillna(0, inplace=True)
    return X_test_estimated_a_processed, X_test_estimated_b_processed, X_test_estimated_c_processed



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


