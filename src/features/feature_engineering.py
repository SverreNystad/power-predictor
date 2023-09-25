from datetime import datetime
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def create_time_features_from_date(data_frame: pd.DataFrame) -> pd.DataFrame:
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
    df = data_frame.copy()
    df["month"] = df["date_forecast"].apply(get_month)
    df["season"] = df["month"].apply(get_season)
    df["year"] = df["date_forecast"].apply(get_year)
    df["day_of_year"] = df["date_forecast"].apply(get_day_of_year)
    df["day_segment"] = df["date_forecast"].apply(get_day_segment)
    return df


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


def get_year(date: datetime) -> int:
    return date.year


def get_day_of_year(date: datetime) -> int:
    return date.timetuple().tm_yday


def get_day_segment(date: datetime) -> int:
    """
    Returns a segment of the day based on the hour and minute of the given date.
    The day is divided into 96 segments, each segment representing a 15-minute interval.
    """
    hour_in_15_min_intervals = date.hour * 4
    minute_in_15_min_intervals = date.minute // 15
    return hour_in_15_min_intervals + minute_in_15_min_intervals


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


def remove_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    df = data_frame.drop("snow_density:kgm3", axis=1)
    df["cloud_coverage_over_50%"] = np.where(
        (df["ceiling_height_agl:m"] >= 0) & (df["ceiling_height_agl:m"] <= 6000), 1, 0
    )
    df = df.drop("ceiling_height_agl:m", axis=1)
    df["no_clouds"] = df["cloud_base_agl:m"].isna().astype(int)
    df["cloud_base_agl:m"] = df["cloud_base_agl:m"].fillna(0)
    return df


def feature_engineer(data_frame: pd.DataFrame) -> pd.DataFrame:
    df = remove_features(data_frame)
    df = create_time_features_from_date(df)
    return df


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

    # Feature engineer
    train_observed_clean = feature_engineer(train_observed)
    train_estimated_clean = feature_engineer(train_estimated)

    # Handle missing values (e.g., imputation, removal)
    train_observed_clean = train_observed.dropna()
    train_estimated_clean = train_estimated.dropna()

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
