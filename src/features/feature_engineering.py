from datetime import datetime
import pandas as pd


def get_month(date: datetime) -> int:
    return date.month

def get_season(month: int) -> int:
    if month == 12:
        return 1
    return ((month) // 3) + 1

def get_year(date: datetime) -> int:
    return date.year

def get_day_of_year(date: datetime) -> int:
    return date.timetuple().tm_yday


def get_day_segment(date: datetime) -> int:
    return date.hour * 4 + date.minute // 15

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

