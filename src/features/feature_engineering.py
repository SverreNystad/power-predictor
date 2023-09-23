from datetime import datetime
import pandas as pd


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
    df["day_of_year"]= df["date_forecast"].apply(get_day_of_year)
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

