from datetime import datetime
import pandas as pd
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import math


def prepare_data(
    train_observed: pd.DataFrame,
    train_estimated: pd.DataFrame,
    test_size=0.2,
    random_state=42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Prepares the data for modeling by handling missing values and splitting the data.

    Args:
    train_observed (pd.DataFrame): The aligned training DataFrame with observed features.
    train_estimated (pd.DataFrame): The aligned training DataFrame with estimated features.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    X_train_obs (pd.DataFrame): The training features with observed data.
    X_val_obs (pd.DataFrame): The validation features with observed data.
    y_train_obs (pd.Series): The training target with observed data.
    y_val_obs (pd.Series): The validation target with observed data.
    X_train_est (pd.DataFrame): The training features with estimated data.
    X_val_est (pd.DataFrame): The validation features with estimated data.
    y_train_est (pd.Series): The training target with estimated data.
    y_val_est (pd.Series): The validation target with estimated data.
    """

    print(f"Before dropping {train_observed.shape}")

    # Remove missing features

    train_observed = remove_missing_features(train_observed)
    train_estimated = remove_missing_features(train_estimated)
    print(f"Description missing values: {train_observed.isna().sum()}")

    # Handle missing values (e.g., imputation, removal)
    train_observed_clean = train_observed.dropna()
    train_estimated_clean = train_estimated.dropna()

    print(f"After dropping {train_observed_clean.shape}")

    # # Feature engineer
    train_observed_clean = feature_engineer(train_observed_clean)
    train_estimated_clean = feature_engineer(train_estimated_clean)

    # Split the data into features (X) and target (y)
    X_obs = train_observed_clean.drop(
        columns=["time", "pv_measurement", "date_forecast"]
    )
    y_obs = train_observed_clean["pv_measurement"]

    X_est = train_estimated_clean.drop(
        columns=["time", "pv_measurement", "date_forecast"]
    )
    y_est = train_estimated_clean["pv_measurement"]

    # Split the data into training and validation sets
    X_train_obs, X_val_obs, y_train_obs, y_val_obs = train_test_split(
        X_obs, y_obs, test_size=test_size, random_state=random_state
    )
    X_train_est, X_val_est, y_train_est, y_val_est = train_test_split(
        X_est, y_est, test_size=test_size, random_state=random_state
    )

    return (
        X_train_obs,
        X_val_obs,
        y_train_obs,
        y_val_obs,
        X_train_est,
        X_val_est,
        y_train_est,
        y_val_est,
    )


def remove_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("snow_density:kgm3", axis=1)
    df = df.drop("ceiling_height_agl:m", axis=1)
    df["cloud_base_agl:m"] = df["cloud_base_agl:m"].fillna(0)
    df = df.drop("elevation:m", axis=1)
    return df


def feature_engineer(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = create_time_features_from_date(data_frame)
    return data_frame


def create_time_features_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new data frame with new features from date_forecast column.
    This will create temporal features from date_forecast that are easier to learn by the model.
    It creates the following features: month, season, year, day_of_year, day_segment.
    All of the new features are int type.

    Args:
        df (pd.DataFrame): Data frame with date_forecast column.
    Returns:
        pd.DataFrame: Data frame copy with new features.

    """
    df["sin_day_of_year"] = df["date_forecast"].apply(get_sin_day)
    df["cos_day_of_year"] = df["date_forecast"].apply(get_cos_day)
    df["sin_hour"] = df["date_forecast"].apply(get_sin_hour)
    df["cos_hour"] = df["date_forecast"].apply(get_cos_hour)
    return df


def get_sin_hour(date: datetime) -> float:
    return math.sin(2 * math.pi * (date.hour) / 24)


def get_cos_hour(date: datetime) -> float:
    return math.cos(2 * math.pi * (date.hour) / 24)


def get_month(date: datetime) -> int:
    return date.month


def get_season(month: int) -> int:
    """
    Returns the season based on the given month. The seasons are divided as follows:
    * Winter: December, January, February (1, 2, 3)
    * Spring: March, April, May (4, 5, 6)
    * Summer: June, July, August (7, 8, 9)
    * Fall: September, October, November (10, 11, 12)
    """
    if month == 12:
        return 1
    return ((month) // 3) + 1


def get_day_of_year(date: datetime) -> int:
    return date.timetuple().tm_yday


def get_sin_day(date: datetime) -> float:
    return math.sin(2 * math.pi * (date.timetuple().tm_yday - 1) / 365.25)


def get_cos_day(date: datetime) -> float:
    return math.cos(2 * math.pi * (date.timetuple().tm_yday - 1) / 365.25)


def get_day_segment(date: datetime) -> int:
    """
    Returns a segment of the day based on the hour and minute of the given date.
    The day is divided into 96 segments, each segment representing a 15-minute interval.
    """
    hour_in_15_min_intervals = date.hour * 4
    minute_in_15_min_intervals = date.minute // 15
    return hour_in_15_min_intervals + minute_in_15_min_intervals


def create_polynomial_features(
    df, columns, degree=2, include_bias=False, interaction_only=False
):
    """
    Create polynomial features for specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names for which to create polynomial features.
    - degree (int): The degree of the polynomial features. Default is 2.
    - include_bias (bool): Whether to include a bias column in the output. Default is False.
    - interaction_only (bool): Whether to include only interaction features. Default is False.

    Returns:
    - df_poly (pd.DataFrame): DataFrame with original and new polynomial features.
    """
    poly = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    )
    poly_features = poly.fit_transform(df[columns])
    feature_names = poly.get_feature_names(input_features=columns)

    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    df_poly = df_poly.drop(
        columns=columns
    )  # Drop original columns as they are included in the polynomial features

    # Concatenate the original DataFrame with the new polynomial features DataFrame
    df_poly = pd.concat([df.drop(columns=columns), df_poly], axis=1)

    return df_poly


def log_transform(df: pd.DataFrame, columns: List[str]):
    df_transformed = df.copy()
    for column in columns:
        df_transformed[column] = np.log1p(df_transformed[column])
    return df_transformed


def scale_features(df: pd.DataFrame, columns: List[str], method="standard"):
    df_scaled = df.copy()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled


def trig_transform(df: pd.DataFrame, column: str, period: int):
    df_trig = df.copy()
    df_trig[f"{column}_sin"] = np.sin(2 * math.pi * df_trig[column] / period)
    df_trig[f"{column}_cos"] = np.cos(2 * math.pi * df_trig[column] / period)
    return df_trig


def create_lagged_features(df, column, n_lags):
    """
    Creates lagged features for a given column in a DataFrame.
    Decide on the column for which you want to create lagged features. Usually, it's the column containing time-series data, like solar energy production or temperature.
    Decide on the number of lagged features (n_lags) you want to create. For example, if n_lags=3, you will create three new columns containing the values of the original column shifted by 1, 2, and 3 time steps, respectively.
    Call the function with the appropriate arguments.

    Args:
        df is the DataFrame containing your time-series data.
        column is the name of the column for which you want to create lagged features.
        n_lags is the number of lagged features you want to create
    """
    for lag in range(1, n_lags + 1):
        df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df


def calculate_rolling_statistics(df, column, window_size):
    df[f"{column}_rolling_mean_{window_size}"] = (
        df[column].rolling(window=window_size).mean()
    )
    df[f"{column}_rolling_std_{window_size}"] = (
        df[column].rolling(window=window_size).std()
    )
    return df


# Define a function to identify positively skewed numerical features in a DataFrame
def identify_skewed_features(df, skew_threshold=1):
    # Select numerical features
    num_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Calculate skewness for each numerical feature and filter those that are positively skewed
    skewed_features = [
        feature for feature in num_features if skew(df[feature]) > skew_threshold
    ]

    return skewed_features


def create_domain_specific_features(df):
    """
    Create domain-specific features for solar energy production.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - df_domain (pd.DataFrame): DataFrame with original and new domain-specific features.
    """
    df_domain = df.copy()

    # Create a binary feature representing whether the sky is clear
    df_domain["is_clear_sky"] = (df_domain["clear_sky_energy_1h:J"] > 0).astype(int)

    # Create a feature representing total sun exposure
    df_domain["total_sun_exposure"] = (
        df_domain["direct_rad:W"] + df_domain["diffuse_rad:W"]
    )

    # Create a binary feature representing whether it is raining
    df_domain["is_raining"] = (df_domain["precip_5min:mm"] > 0).astype(int)

    # Create a binary feature representing whether there is snow cover
    df_domain["is_snow_cover"] = (df_domain["snow_depth:cm"] > 0).astype(int)

    return df_domain


def remove_date_time_feature(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove date_forecast column from data frame.

    Args:
        df (pd.DataFrame): Data frame with date_forecast column.
    Returns:
        pd.DataFrame: Data frame copy without date_forecast column.

    """
    df = data_frame.copy()
    df.drop(["date_forecast"], axis=1, inplace=True)
    return df


def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data frame by removing outliers and NaN values.

    Args:
        df (pd.DataFrame): Data frame with date_forecast column.
    Returns:
        pd.DataFrame: Data frame copy without outliers and NaN values.

    """
    df = data_frame.copy()
    df = create_time_features_from_date(df)
    # df = df.dropna()
    df = df[df["target"] > 0]
    return df


def add_location(data_frame: pd.DataFrame, location: str):
    if location.lower() == "a":
        data_frame["location_a"] = 1
    else:
        data_frame["location_a"] = 0

    if location.lower() == "b":
        data_frame["location_b"] = 1
    else:
        data_frame["location_b"] = 0

    if location.lower() == "c":
        data_frame["location_c"] = 1
    else:
        data_frame["location_c"] = 0
    return data_frame


# Define a function to align the temporal resolution of the datasets
def temporal_alignment(
    train: pd.DataFrame, observed: pd.DataFrame, estimated: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns the temporal resolution of the datasets by aggregating the 15-min interval weather data to hourly intervals.

    Args:
        train (pd.DataFrame): The training targets DataFrame.
        observed (pd.DataFrame): The observed training features DataFrame.
        estimated (pd.DataFrame): The estimated training features DataFrame.

    Returns:
        train_observed (pd.DataFrame): The aligned training DataFrame with observed features.
        train_estimated (pd.DataFrame): The aligned training DataFrame with estimated features.
    """
    # Convert the time columns to datetime objects
    train["time"] = pd.to_datetime(train["time"])
    observed["date_forecast"] = pd.to_datetime(observed["date_forecast"])
    estimated["date_forecast"] = pd.to_datetime(estimated["date_forecast"])

    # Set the date_forecast column as index for resampling
    observed.set_index("date_forecast", inplace=True)
    estimated.set_index("date_forecast", inplace=True)

    # Resample the weather data to hourly intervals and aggregate the values by mean
    observed_resampled = observed.resample("1H").mean()
    estimated_resampled = estimated.resample("1H").mean()

    # Reset the index after resampling
    observed_resampled.reset_index(inplace=True)
    estimated_resampled.reset_index(inplace=True)

    # Merge the aggregated weather data with the solar production data based on the timestamp
    train_observed = pd.merge(
        train, observed_resampled, how="left", left_on="time", right_on="date_forecast"
    )
    train_estimated = pd.merge(
        train, estimated_resampled, how="left", left_on="time", right_on="date_forecast"
    )

    return train_observed, train_estimated


def _add_calc_date_and_correct_target(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add date_calc feature for each hour of the next day and now

    Args:
        data_frame (pd.DataFrame): Data frame with date_forecast column.
    Returns:
        pd.DataFrame: Data frame copy with date_calc column and corresponding pv_measurements.
    """

    # Check that the date_calc is not already in the dataframe
    if "date_calc" in data_frame.columns:
        return data_frame

    df = data_frame.copy()

    # Add date_calc for current time so that the model can learn what weather conditions lead to the current pv_measurement
    df["date_calc"] = df["date_forecast"] + pd.DateOffset(hours=0)

    # Add date_calc for each hour of the next day
    rows_list = []
    start_of_next_day = 24
    end_of_next_day = 48

    pv_lookup = df.set_index("date_forecast")["pv_measurement"].to_dict()

    rows_list = []

    # Vectorized approach to create new rows
    for next_day_hour in range(
        start_of_next_day, end_of_next_day
    ):  # 25 to 48 hours for the next day
        new_rows = df.copy()

        new_rows["date_forecast"] = df["date_forecast"] + pd.to_timedelta(
            next_day_hour, unit="h"
        )
        # Lookup the pv_measurement for the corresponding date_forecast
        new_rows["pv_measurement"] = new_rows["date_forecast"].map(pv_lookup)
        new_rows["date_forecast"] = next_day_hour
        rows_list.append(new_rows)

    df_new_rows = pd.concat(rows_list, ignore_index=True)
    df = (
        pd.concat([df, df_new_rows], ignore_index=True)
        .sort_values(by=["date_calc"])
        .reset_index(drop=True)
    )

    return df
