from src.features.preprocess_data import fetch_preprocessed_data, get_preprocessed_test_data


def test_fetch_preprocessed_data_pairs_pv_to_correct_x_date():
    """ Make sure that the pv_measurement is the same for corresponding as the date_forecast"""
    # Arrange
    X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined = fetch_preprocessed_data(drop_features=False)
    
    # Manually checked
    # Observed
    # A 2022-07-01 06:00:00, pv 1997.8200000000004
    # B 2019-07-21 14:00:00, pv 207.8625
    # C 2021-07-02 12:00:00, pv 813.4000000000001


    # Test estimated data
    # A 2023-03-03 14:00:00, pv 271.48
    # B 2023-02-08 03:00:00,  pv -0.0
    # C 2022-11-25 17:00:00, pv 0.0
    
    # Check all the dates are the same
    assert X_train_obs_combined["pv_measurement"] == y_train_obs_combined
    assert X_val_obs_combined["pv_measurement"] == y_val_obs_combined
    assert X_train_est_combined["pv_measurement"] == y_train_est_combined
    assert X_val_est_combined["pv_measurement"] == y_val_est_combined

def test_fetch_preprocessed_data_gives_data():
    X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined = fetch_preprocessed_data()
    
    # Check that the data is not empty
    assert X_train_obs_combined.shape[0] > 0
    assert X_val_obs_combined.shape[0] > 0
    assert y_train_obs_combined.shape[0] > 0
    assert y_val_obs_combined.shape[0] > 0
    assert X_train_est_combined.shape[0] > 0
    assert X_val_est_combined.shape[0] > 0
    assert y_train_est_combined.shape[0] > 0
    assert y_val_est_combined.shape[0] > 0

    # Check that the x and y data have the same number of rows
    assert X_train_obs_combined.shape[0] == y_train_obs_combined.shape[0]
    assert X_val_obs_combined.shape[0] == y_val_obs_combined.shape[0]
    assert X_train_est_combined.shape[0] == y_train_est_combined.shape[0]
    assert X_val_est_combined.shape[0] == y_val_est_combined.shape[0]

def test_fetch_preprocessed_data_gives_same_number_of_features():
    X_train_obs_combined, X_val_obs_combined, _, _, X_train_est_combined, X_val_est_combined, _, _ = fetch_preprocessed_data()
    
    # Check that the data has the same number of features
    assert X_train_obs_combined.shape[1] == X_val_obs_combined.shape[1] == X_train_est_combined.shape[1] == X_val_est_combined.shape[1]
    

def test_get_preprocessed_test_data_has_same_features_as_training():
    # Arrange
    X_train_obs_combined, X_val_obs_combined, _, _, X_train_est_combined, X_val_est_combined, _, _ = fetch_preprocessed_data()
    x_test_whole = get_preprocessed_test_data()
    
    # Check that the test data has the same number of features as the training data
    assert x_test_whole.shape[1] == X_train_obs_combined.shape[1] == X_val_obs_combined.shape[1] == X_train_est_combined.shape[1] == X_val_est_combined.shape[1]

def test_get_preprocessed_test_data_has_no_missing_values():
    # Arrange
    x_test_whole = get_preprocessed_test_data()
    
    # Check that the test data has no missing values
    assert x_test_whole.isnull().sum().sum() == 0

def test_get_preprocessed_test_data_has_same_amount_of_rows_as_final_test():
    # Arrange
    x_test_whole = get_preprocessed_test_data()
    SUBMISSION_LENGTH = 2160
    
    # Check that the test data has the same number of rows as the final test data
    assert x_test_whole.shape[0] == SUBMISSION_LENGTH

def test_get_preprocessed_test_data_has_correct_amount_from_each_location():
    # Arrange
    x_test_whole = get_preprocessed_test_data()
    SUBMISSION_LENGTH = 2160
    
    x_test_whole_a = x_test_whole[x_test_whole["location_a"] == 1]
    x_test_whole_b = x_test_whole[x_test_whole["location_b"] == 1]
    x_test_whole_c = x_test_whole[x_test_whole["location_c"] == 1]

    # Check that they have the correct amount of rows
    assert x_test_whole_a.shape[0] == SUBMISSION_LENGTH/3
    assert x_test_whole_b.shape[0] == SUBMISSION_LENGTH/3
    assert x_test_whole_c.shape[0] == SUBMISSION_LENGTH/3
    