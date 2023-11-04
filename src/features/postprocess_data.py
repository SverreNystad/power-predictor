import pandas as pd
import math

from src.features.preprocess_data import fetch_preprocessed_data

find_time_sin = lambda hour: math.sin(2 * math.pi * (hour) / 24)
find_time_cos = lambda hour: math.cos(2 * math.pi * (hour) / 24)

def postprocess_data(x_test: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the data to set the predicted values to 0 at the correct times."""
    
    # Cap the min and max values for each location for each hour
    y_pred = cap_min_max_values(x_test, y_pred)

    # Set the predicted values to 0 at the correct times
    y_pred = set_0_pv_at_times(x_test, y_pred, "a", [22, 23, 0])
    y_pred = set_0_pv_at_times(x_test, y_pred, "b", [22, 23, 0])
    y_pred = set_0_pv_at_times(x_test, y_pred, "c", [22, 23, 0])

    return y_pred

def cap_min_max_values(x_test: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Cap the min and max values for each location for each hour."""
    for hour in range(24):
        # Get the min and max values for each location for each hour
        min_value_a, max_value_a = get_min_max_values_for_location_at_hour("a", hour)
        min_value_b, max_value_b = get_min_max_values_for_location_at_hour("b", hour)
        min_value_c, max_value_c = get_min_max_values_for_location_at_hour("c", hour)
        print(f"hour: {hour}, min_value_a: {min_value_a}, max_value_a: {max_value_a}, min_value_b: {min_value_b}, max_value_b: {max_value_b}, min_value_c: {min_value_c}, max_value_c: {max_value_c}")
        # Cap the values between min_value and max_value
        y_pred = cap_min_max_values_for_hour(x_test, y_pred, "a", hour, min_value_a, max_value_a)
        y_pred = cap_min_max_values_for_hour(x_test, y_pred, "b", hour, min_value_b, max_value_b)
        y_pred = cap_min_max_values_for_hour(x_test, y_pred, "c", hour, min_value_c, max_value_c)
    return y_pred

X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined = fetch_preprocessed_data(drop_features=False)
x_whole_with_time = pd.concat([X_train_obs_combined, X_val_obs_combined, X_train_est_combined, X_val_est_combined])

def get_min_max_values_for_location_at_hour(location: str, hour: int) -> tuple[float, float]:
    """Get the min and max values for a specific location at a specific hour."""
    # Get the x and y for the given hour and location
    hour_sin = find_time_sin(hour)
    hour_cos = find_time_cos(hour)
    # find the min and max values for the given hour and location
    min_value = x_whole_with_time[(x_whole_with_time["location_" + location] == 1) & (x_whole_with_time["sin_hour"] == hour_sin) & (x_whole_with_time["cos_hour"] == hour_cos)]["pv_measurement"].min()
    max_value = x_whole_with_time[(x_whole_with_time["location_" + location] == 1) & (x_whole_with_time["sin_hour"] == hour_sin) & (x_whole_with_time["cos_hour"] == hour_cos)]["pv_measurement"].max()
    
    return (min_value, max_value)

def cap_min_max_values_for_hour(x_test: pd.DataFrame, y_pred: pd.DataFrame, location: str, hour: int, min_value: float, max_value: float) -> pd.DataFrame:
    """Cap the min and max values for a specific hour."""
    
    # Calculate sin and cos values for the given hour
    hour_sin = find_time_sin(hour)
    hour_cos = find_time_cos(hour)
    
    # Find indices corresponding to the given hour at the given location
    indices = x_test[(x_test["location_" + location] == 1) & (x_test["sin_hour"] == hour_sin) & (x_test["cos_hour"] == hour_cos)].index
    
    # Cap the values between min_value and max_value
    y_pred.loc[indices] = y_pred.loc[indices].clip(min_value, max_value)
    
    return y_pred

def set_0_pv_at_times(x_test: pd.DataFrame, y_pred: pd.DataFrame, location: str, hours: list[int]) -> pd.DataFrame:
    """Find the correct predicted values at the given times and locaiton and set them to 0."""
    hours_to_set_0_sin = [find_time_sin(hour) for hour in hours]
    hours_to_set_0_cos = [find_time_cos(hour) for hour in hours]


    indices = x_test[(x_test["location_" + location] == 1) & (x_test["sin_hour"].isin(hours_to_set_0_sin) & (x_test["cos_hour"].isin(hours_to_set_0_cos)))].index
    for index in indices:
        y_pred.loc[index] = 0
    return y_pred
