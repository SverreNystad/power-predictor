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

    # Remove missing features
    train_observed = remove_missing_features(train_observed)
    train_estimated = remove_missing_features(train_estimated)

    # Handle missing values (e.g., imputation, removal)
    train_observed_clean = train_observed.dropna(
        subset=["visibility:m", "pv_measurement"]
    )
    train_estimated_clean = train_estimated.dropna(
        subset=["visibility:m", "pv_measurement"]
    )

    # Remove discrepancies
    train_observed_clean = remove_discrepancies(train_observed_clean)
    train_estimated_clean = remove_discrepancies(train_estimated_clean)

    # Feature engineer
    train_observed_clean = feature_engineer(train_observed_clean)
    train_estimated_clean = feature_engineer(train_estimated_clean)

    # Split the data into features (X) and target (y)
    X_obs = train_observed_clean.drop(
        columns=["time", "pv_measurement", "date_forecast", "date_calc"] , errors="ignore"
    )
    y_obs = train_observed_clean["pv_measurement"]

    X_est = train_estimated_clean.drop(
        columns=["time", "pv_measurement", "date_calc", "date_forecast"]
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


def get_location_datasets(
    df: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    locations = ["location_a", "location_b", "location_c"]
    x_a = df[df["location_a"] == 1]
    x_a = x_a.drop(locations, axis=1)
    y_a = x_a["pv_measurement"]
    if "pv_measurement" in x_a.columns:
        x_a = x_a.drop("pv_measurement", axis=1)

    x_b = df[df["location_b"] == 1]
    x_b = x_b.drop(locations, axis=1)
    y_b = x_b["pv_measurement"]
    if "pv_measurement" in x_b.columns:
        x_b = x_b.drop("pv_measurement", axis=1)

    x_c = df[df["location_c"] == 1]
    x_c = x_c.drop(locations, axis=1)
    y_c = x_c["pv_measurement"]
    if "pv_measurement" in x_b.columns:
        x_b = x_b.drop("pv_measurement", axis=1)

    return (x_a, x_b, x_c, y_a, y_b, y_c)


def remove_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("snow_density:kgm3", axis=1)
    df = df.drop("ceiling_height_agl:m", axis=1)
    df["cloud_base_agl:m"] = df["cloud_base_agl:m"].fillna(0)
    df = df.drop("elevation:m", axis=1)
    return df


def remove_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_positive_pv_in_night(df)
    df = remove_night_light_discrepancies(df)
    df = remove_zero_value_discrepancies(df)
    df = remove_faulty_zero_measurements_for_direct_sun_light(df)
    # df = remove_outliers(df)
    return df


def remove_positive_pv_in_night(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove positive pv measurements when is_day is 0 and pv_measurement is positive and pv_measurement is the same next timestep
    """
    # Remove positive pv measurements when is_day is 0 and pv_measurement is positive and pv_measurement is the same next timestep
    df = df.drop(
        df[
            (df["is_day:idx"] == 0)
            & (df["pv_measurement"] > 0)
            & (df["pv_measurement"] == df["pv_measurement"].shift(1))
        ].index
    )

    # Remove positive pv measurements when sun_elevation is negative
    threshold = -10
    df = df.drop(df[(df["sun_elevation:d"] < threshold) & (df["pv_measurement"] > 0)].index)
    return df



def remove_outliers(df: pd.DataFrame, lower_bound: float = 0.1, upper_bound: float = 0.9) -> pd.DataFrame:
    '''
    Removing outliers using IQR method
    '''

    columns_to_check = [col for col in df.columns if col != "pv_measurement"]
    for col in columns_to_check:
        # Calculate IQR
        Q1 = df[col].quantile(lower_bound)
        Q3 = df[col].quantile(upper_bound)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def remove_night_light_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    # Remove all rows where pv_measurement has the same value for 6 timesteps and not is 0 remove them

    # Step 1: Identify runs of equal, non-zero values
    df["group"] = (
        (df["pv_measurement"] != df["pv_measurement"].shift())
        | (df["pv_measurement"] == 0)
    ).cumsum()

    # Step 2: Count occurrences in each run
    counts = df.groupby("group")["pv_measurement"].transform("count")

    # Step 3: Identify groups to remove
    to_remove = (counts >= 6) & (df["pv_measurement"] != 0)

    # Step 4: Remove those rows
    df_cleaned = df[~to_remove].drop(columns=["group"])
    return df_cleaned


def remove_zero_value_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    # Remove all rows where pv_measurement has the same value for 100 timesteps and is 0 remove them

    # Didn't do anything lol

    # Step 1: Identify runs of equal, non-zero values
    return df
    df["group"] = (
        (df["pv_measurement"] != df["pv_measurement"].shift())
        | (df["pv_measurement"] == 0)
    ).cumsum()

    # Step 2: Count occurrences in each run
    counts = df.groupby("group")["pv_measurement"].transform("count")

    # Step 3: Identify groups to remove
    to_remove = (counts >= 50) & (df["pv_measurement"] == 0) & (df["is_day:idx"] == 1)

    # Step 4: Remove those rows
    df_cleaned = df[~to_remove].drop(columns=["group"])
    return df_cleaned


def remove_faulty_zero_measurements_for_direct_sun_light(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """ """
    mask = ((df["diffuse_rad:W"] + df["direct_rad:W"]) >= 30) & (
        df["pv_measurement"] == 0
    )
    df = df[~mask]
    return df


def feature_engineer(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = create_time_features_from_date(data_frame)
    # data_frame = create_expected_pv_based_on_previous_years_same_day(data_frame)
    data_frame["sun_product"] = data_frame["diffuse_rad:W"] * data_frame["direct_rad:W"]

    data_frame["modified_solar_elevation"] = np.where(
        data_frame["sun_elevation:d"] <= 0,
        0,
        np.sin(np.radians(data_frame["sun_elevation:d"])),
    )
    data_frame = data_frame.drop("sun_elevation:d", axis=1)

    data_frame["effective_radiation"] = np.where(
        data_frame["clear_sky_energy_1h:J"] == 0,
        0,  # or your specified value
        data_frame["direct_rad_1h:J"] / data_frame["clear_sky_energy_1h:J"],
    )

    # Check for the existence of date_calc column
    if "date_calc" not in data_frame.columns:
        data_frame["time_since_prediction"] = 0
    else:
        data_frame["time_since_prediction"] = (
            data_frame["date_forecast"] - data_frame["date_calc"]
        ).dt.total_seconds() / 3600

    # data_frame["time_since_prediction"] =

    # data_frame["residual_radiation"] = (
    #     data_frame["clear_sky_rad:W"]
    #     - data_frame["direct_rad:W"]
    #     - data_frame["diffuse_rad:W"]
    # )

    # WAS WORSE
    # data_frame["effective_radiation2"] = np.where(
    #     data_frame["clear_sky_rad:W"] == 0,
    #     0,  # or your specified value
    #     data_frame["direct_rad:W"] / data_frame["clear_sky_rad:W"],
    # )

    data_frame["cloud_ratio"] = np.where(
        data_frame["total_cloud_cover:p"] == 0,
        0,  # or your specified value
        data_frame["effective_cloud_cover:p"] / data_frame["total_cloud_cover:p"],
    )

    # data_frame["diffuse_cloud_conditional_interaction"] = data_frame[
    #     "diffuse_rad:W"
    # ].where(data_frame["effective_cloud_cover:p"] < 0.3, 0)

    data_frame["cloud_cover_over_30%"] = np.where(
        data_frame["effective_cloud_cover:p"] > 30, 1, 0
    )

    snow_columns = [
        "snow_depth:cm",
        "fresh_snow_12h:cm",
        "fresh_snow_1h:cm",
        "fresh_snow_24h:cm",
        "fresh_snow_3h:cm",
        "fresh_snow_6h:cm",
    ]

    # data_frame["surface_temperature"] = data_frame.apply(calculate_surface_temp, axis=1)
    # data_frame["50m_temperature"] = data_frame.apply(calculate_50m_temp, axis=1)
    # data_frame["100m_temperature"] = data_frame.apply(calculate_100m_temp, axis=1)
    # data_frame = data_frame.drop("t_1000hPa:K", axis=1)

    # data_frame["sun_addition"] = (
    #     data_frame["direct_rad_1h:J"] + data_frame["diffuse_rad_1h:J"]
    # )

    # Create a feature that is precip_5min:mm * precip_type_5min.idx
    # data_frame["any_precip"] = (
    #     data_frame["precip_5min:mm"] * data_frame["precip_type_5min:idx"]
    # )

    data_frame["sun_addition"] = (
        data_frame["diffuse_rad:W"] + data_frame["direct_rad:W"]
    )

    # data_frame["sun addition W to 1h ratio"] = (
    #     data_frame["sun_addition"]
    #     * 3600
    #     / (data_frame["direct_rad_1h:J"] + data_frame["diffuse_rad_1h:J"])
    # ).fillna(1)

    data_frame["is_freezing"] = (data_frame["t_1000hPa:K"] < 273).astype(int)

    data_frame["is_snow"] = (data_frame[snow_columns] > 0).any(axis=1).astype(int)
    data_frame["is_rain"] = (data_frame["precip_5min:mm"] > 0).astype(int)

    data_frame = data_frame.drop("snow_drift:idx", axis=1)
    data_frame = data_frame.drop("snow_depth:cm", axis=1)
    data_frame = data_frame.drop("snow_water:kgm2", axis=1)
    data_frame = data_frame.drop("fresh_snow_12h:cm", axis=1)
    data_frame = data_frame.drop("fresh_snow_1h:cm", axis=1)
    data_frame = data_frame.drop("fresh_snow_24h:cm", axis=1)
    data_frame = data_frame.drop("fresh_snow_3h:cm", axis=1)
    data_frame = data_frame.drop("fresh_snow_6h:cm", axis=1)
    data_frame = data_frame.drop("snow_melt_10min:mm", axis=1)

    data_frame = data_frame.drop("msl_pressure:hPa", axis=1)
    data_frame = data_frame.drop("pressure_100m:hPa", axis=1)
    data_frame = data_frame.drop("sfc_pressure:hPa", axis=1)

    # Add maximum pv_measurement based on location
    # max_pv_a = 5733.42
    # max_pv_b = 1152.3
    # max_pv_c = 999.6
    # data_frame["max_pv_location"] = np.where(
    #     data_frame["location_a"] == 1,
    #     max_pv_a,
    #     np.where(
    #         data_frame["location_b"] == 1,
    #         max_pv_b,
    #         np.where(data_frame["location_c"] == 1, max_pv_c, np.nan),
    #     ),
    # )

    # # Add average pv_measurement based on location
    # average_pv_a = 814.88
    # average_pv_b = 129.375
    # average_pv_c = 117.6
    # data_frame["average_pv_location"] = np.where(
    #     data_frame["location_a"] == 1,
    #     average_pv_a,
    #     np.where(
    #         data_frame["location_b"] == 1,
    #         average_pv_b,
    #         np.where(data_frame["location_c"] == 1, average_pv_c, np.nan),
    #     ),
    # )

    # # Add Maximum_pv_location times sun_addition
    # data_frame["max_pv_location_times_sun_addition"] = (
    #     data_frame["max_pv_location"] * data_frame["sun_addition"]
    # )

    # # Add Maximum_pv_location times sun_addition
    # data_frame["average_pv_location_times_sun_addition"] = (
    #     data_frame["average_pv_location"] * data_frame["sun_addition"]
    # )

    # data_frame["sun_addition_temp_ratio"] = (
    #     data_frame["max_pv_location"]
    #     * data_frame["sun_addition"]
    #     / data_frame["t_1000hPa:K"]
    # )

    return data_frame


def calculate_surface_temp(row):
    # Constants
    R = 287  # Specific gas constant for dry air, J kg^-1 K^-1
    g = 9.81  # Acceleration due to gravity, m s^-2
    lapse_rate = 0.0065  # Average lapse rate, K m^-1

    P1 = 1000  # Initial pressure level, hPa
    P2 = row["sfc_pressure:hPa"]  # Final pressure level, hPa
    T1 = row["t_1000hPa:K"]  # Temperature at P1, K

    # Altitude difference using barometric formula (approximation)
    delta_h = (R * T1 / g) * np.log(P1 / P2)

    # Temperature difference using constant lapse rate
    delta_T = lapse_rate * delta_h

    # Temperature at P2, converting from K to C
    T2_C = T1 - delta_T - 273.15

    return T2_C


def calculate_50m_temp(row):
    # Constants
    R = 287  # Specific gas constant for dry air, J kg^-1 K^-1
    g = 9.81  # Acceleration due to gravity, m s^-2
    lapse_rate = 0.0065  # Average lapse rate, K m^-1

    P1 = 1000  # Initial pressure level, hPa
    P2 = row["pressure_50m:hPa"]  # Final pressure level, hPa
    T1 = row["t_1000hPa:K"]  # Temperature at P1, K

    # Altitude difference using barometric formula (approximation)
    delta_h = (R * T1 / g) * np.log(P1 / P2)

    # Temperature difference using constant lapse rate
    delta_T = lapse_rate * delta_h

    # Temperature at P2, converting from K to C
    T2_C = T1 - delta_T - 273.15

    return T2_C


def calculate_100m_temp(row):
    # Constants
    R = 287  # Specific gas constant for dry air, J kg^-1 K^-1
    g = 9.81  # Acceleration due to gravity, m s^-2
    lapse_rate = 0.0065  # Average lapse rate, K m^-1

    P1 = 1000  # Initial pressure level, hPa
    P2 = row["pressure_100m:hPa"]  # Final pressure level, hPa
    T1 = row["t_1000hPa:K"]  # Temperature at P1, K

    # Altitude difference using barometric formula (approximation)
    delta_h = (R * T1 / g) * np.log(P1 / P2)

    # Temperature difference using constant lapse rate
    delta_T = lapse_rate * delta_h

    # Temperature at P2, converting from K to C
    T2_C = T1 - delta_T - 273.15

    return T2_C


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
    HOURS_OF_DAY = 24
    return math.sin(2 * math.pi * (date.hour) / 24)


def get_cos_hour(date: datetime) -> float:
    HOURS_OF_DAY = 24
    return math.cos(2 * math.pi * (date.hour) / 24)


def get_sin_day(date: datetime) -> float:
    return math.sin(2 * math.pi * (date.timetuple().tm_yday - 1) / 365.25)


def get_cos_day(date: datetime) -> float:
    return math.cos(2 * math.pi * (date.timetuple().tm_yday - 1) / 365.25)


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


def create_expected_pv_based_on_previous_years_same_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mean pv_measurement for each data point based on the previous years same day and hour
    Add this as a feature to the data frame
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least columns: 
                           location_a, location_b, location_c, date_forecast, 
                           pv_measurement, sin_day_of_year, cos_day_of_year, sin_hour, cos_hour
    
    Returns:
        pd.DataFrame: DataFrame with additional feature of mean pv_measurement based on historical data
    """
    df = df.copy()
    # When the data does not contain needed columns, return the original df.
    if not all(
        col in df.columns
        for col in [
            "location_a",
            "location_b",
            "location_c",
            "pv_measurement",
            "sin_day_of_year",
            "cos_day_of_year",
            "sin_hour",
            "cos_hour",
        ]
    ):
        return df
    # Identify the location from the binary flags
    df['location'] = df[['location_a', 'location_b', 'location_c']].idxmax(axis=1)
    
    # Calculate mean pv_measurement for each location, sin_day_of_year, cos_day_of_year, sin_hour, and cos_hour
    mean_pv = df.groupby(['location', 'sin_day_of_year', 'cos_day_of_year', 'sin_hour', 'cos_hour'])['pv_measurement'].mean().reset_index()
    mean_pv.rename(columns={'pv_measurement': 'mean_pv_measurement'}, inplace=True)
    
    # Merge mean_pv_measurement back to the original DataFrame
    df = pd.merge(df, mean_pv, on=['location', 'sin_day_of_year', 'cos_day_of_year', 'sin_hour', 'cos_hour'], how='left')
    df.drop(columns=['location'], inplace=True)
    return df


def create_simple_rolling_mean(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """
    Creates a simple rolling mean feature for a given column in a DataFrame.
    
    Args:
        df: DataFrame containing your time-series data.
        column: The name of the column for which you want to create lagged features.
        window: The size of the window for calculating the rolling mean. 
                For example, if window=10, it will take the previous 10 days.
    """
    # Ensure the DataFrame is sorted by date
    df = df.sort_values(by='date_forecast')
    
    # Ensure 'date_forecast' is in datetime format
    df['date_forecast'] = pd.to_datetime(df['date_forecast'])
    
    # Calculate the rolling mean
    df['rolling_mean_of_' + column] = df[column].rolling(window=window).mean()
    
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


def temporal_alignment_tests(test: pd.DataFrame) -> Tuple[pd.DataFrame]:
    return aggregate_rows(test)


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Create a 'group' column to group every 4 rows together
    df["group"] = df.index // 4

    # Define the aggregation functions
    aggregation = {col: "mean" for col in df.columns if col != "date_forecast"}
    aggregation["date_forecast"] = "first"

    # Group by the 'group' column and aggregate
    df_agg = df.groupby("group").agg(aggregation).reset_index(drop=True)

    # Drop the 'group' column from the original dataframe
    df_agg.drop("group", axis=1, inplace=True)

    return df_agg
