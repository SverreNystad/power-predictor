from datetime import datetime
import pandas as pd
from typing import Tuple
import numpy as np


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
    df = df.dropna()
    df = df[df["target"] > 0]
    return df


def remove_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    df = data_frame.drop("snow_density:kgm3", axis=1)
    df["cloud_coverage_over_50%"] = np.where(
        (df["ceiling_height_agl:m"] >= 0) & (df["ceiling_height_agl:m"] <= 6000), 1, 0
    )
    df = df.drop("ceiling_height_agl:m", axis=1)
    df["no_clouds"] = df["cloud_base_agl:m"].isna().astype(int)
    return df


def prepare_data(
    targets: pd.DataFrame,
    features_observed: pd.DataFrame,
    features_estimated: pd.DataFrame,
    features_test: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Prepare data for modeling.

    Args:
        targets (pd.DataFrame): Data frame with target column.
        features_observed (pd.DataFrame): Data frame with observed features.
        features_estimated (pd.DataFrame): Data frame with estimated features.
        features_test (pd.DataFrame): Data frame with test features.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Tuple containing X_train, y_train, X_val, y_val, X_test, train_data, and val_data DataFrames.
    """

    # Convert the date column to datetime type
    targets["time"] = pd.to_datetime(targets["time"])
    features_observed["date_forecast"] = pd.to_datetime(
        features_observed["date_forecast"]
    )
    features_estimated["date_forecast"] = pd.to_datetime(
        features_estimated["date_forecast"]
    )
    features_test["date_forecast"] = pd.to_datetime(features_test["date_forecast"])

    #  Set the Date Column as Index
    targets.set_index("time", inplace=True)
    features_observed.set_index("date_forecast", inplace=True)
    features_estimated.set_index("date_forecast", inplace=True)
    features_test.set_index("date_forecast", inplace=True)

    # Drop rows with any missing values in targets
    targets.dropna(inplace=True)

    # Downsampling the features to hourly resolution by taking the mean of every 4 rows
    features_observed = features_observed.resample("1H").mean()
    features_estimated = features_estimated.resample("1H").mean()
    features_test = features_test.resample("1H").mean()

    # Merge observed features and targets for training
    train_data = pd.concat([targets, features_observed], axis=1, join="inner").dropna()

    # Merge estimated features and targets for validation
    val_data = pd.concat([targets, features_estimated], axis=1, join="inner").dropna()

    # Define X_train, y_train, X_val, y_val
    X_train = train_data.drop(columns=["pv_measurement"])
    y_train = train_data["pv_measurement"]

    X_val = val_data.drop(columns=["pv_measurement"])
    y_val = val_data["pv_measurement"]

    # For test data, you only have features
    X_test = features_test

    return X_train, y_train, X_val, y_val, X_test, train_data, val_data
