import numpy as np
from src.features.postprocess_data import cap_min_max_values
from src.features.preprocess_data import get_preprocessed_test_data

def test_zero_predictions():
    # Arrange
    x_test_whole = get_preprocessed_test_data()
    test_zero_predictions = np.zeros(x_test_whole.shape[0])
    
    # Act
    post_processed_predictions = cap_min_max_values(x_test_whole, test_zero_predictions)
    
    # Assert
    assert not np.all(post_processed_predictions == 0)

def test_to_large_predictions():
    x_test_whole = get_preprocessed_test_data()
    to_large_value = 100000000
    test_to_large_predictions = np.zeros(x_test_whole.shape[0] + to_large_value)
    
    # Act
    post_processed_predictions = cap_min_max_values(x_test_whole, test_to_large_predictions)
    
    # Assert
    assert np.all(post_processed_predictions < to_large_value)
