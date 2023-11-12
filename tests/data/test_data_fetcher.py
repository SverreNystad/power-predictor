from src.data.data_fetcher import get_raw_data

def test_get_raw_data_correct_shapes():
    # Arrange values like they are from csv
    target_a_shape = (34085, 2)
    target_b_shape = (32848, 2)
    target_c_shape = (32155, 2)

    train_a_obs_shape = (118669, 46)
    train_b_obs_shape = (116929, 46)
    train_c_obs_shape = (116825, 46)

    train_a_est_shape = (17576, 47)
    train_b_est_shape = (17576, 47)
    train_c_est_shape = (17576, 47)


    X_test_shape = (2880, 47)
    # Act
    train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()
    # Assert
    # Training targets
    assert train_a.shape == target_a_shape
    assert train_b.shape == target_b_shape
    assert train_c.shape == target_c_shape

    # Training features observed
    assert X_train_observed_a.shape == train_a_obs_shape
    assert X_train_observed_b.shape == train_b_obs_shape
    assert X_train_observed_c.shape == train_c_obs_shape

    # Training features estimated
    assert X_train_estimated_a.shape == train_a_est_shape
    assert X_train_estimated_b.shape == train_b_est_shape
    assert X_train_estimated_c.shape == train_c_est_shape

    # Test features
    assert X_test_estimated_a.shape == X_test_shape
    assert X_test_estimated_b.shape == X_test_shape
    assert X_test_estimated_c.shape == X_test_shape


def test_first_and_last_are_the_same():
    pass
    

