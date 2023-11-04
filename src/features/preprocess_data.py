from typing import Tuple
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from src.data.data_fetcher import get_raw_data, get_tests
from src.features.feature_engineering import (
    feature_engineer,
    prepare_data,
    remove_missing_features,
    temporal_alignment,
    add_location,
    temporal_alignment_tests,
)


def fetch_preprocessed_data(
    drop_features: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fetch the preprocessed data for training and validation.

    Returns:
        X_train_obs_combined: The observed data for training
        X_val_obs_combined: The observed data for validation
        y_train_obs_combined: The observed labels for training
        y_val_obs_combined: The observed labels for validation
        X_train_est_combined: The estimated data for training
        X_val_est_combined: The estimated data for validation
        y_train_est_combined: The estimated labels for training
        y_val_est_combined: The estimated labels for validation
    """
    (
        train_a,
        train_b,
        train_c,
        X_train_estimated_a,
        X_train_estimated_b,
        X_train_estimated_c,
        X_train_observed_a,
        X_train_observed_b,
        X_train_observed_c,
        _,
        _,
        _,
    ) = get_raw_data()

    # Interpolate pv measurements

    train_b, changes_df1 = interpolate_and_report(train_b)
    train_c, changes_df2 = interpolate_and_report(train_c)

    # Temporally align the data from all three locations to the same time.
    train_observed_a, train_estimated_a = temporal_alignment(
        train_a, X_train_observed_a, X_train_estimated_a
    )
    train_observed_b, train_estimated_b = temporal_alignment(
        train_b, X_train_observed_b, X_train_estimated_b
    )
    train_observed_c, train_estimated_c = temporal_alignment(
        train_c, X_train_observed_c, X_train_estimated_c
    )

    # Add location data
    train_observed_a = add_location(train_observed_a, "a")
    train_estimated_a = add_location(train_estimated_a, "a")

    train_observed_b = add_location(train_observed_b, "b")
    train_estimated_b = add_location(train_estimated_b, "b")

    train_observed_c = add_location(train_observed_c, "c")
    train_estimated_c = add_location(train_estimated_c, "c")

    # Combine the temporally aligned datasets from all three locations
    train_observed_combined = pd.concat(
        [train_observed_a, train_observed_b, train_observed_c], ignore_index=True
    )
    train_estimated_combined = pd.concat(
        [train_estimated_a, train_estimated_b, train_estimated_c], ignore_index=True
    )

    print("train_observed_combined.isna().sum())")
    print(train_observed_combined.isna().sum())
    print("train_estimated_combinedtrain_observed_combined.isna().sum())")
    print(train_estimated_combined.isna().sum())

    # Prepare the combined dataset by handling missing values and splitting the data
    (
        X_train_obs_combined,
        X_val_obs_combined,
        y_train_obs_combined,
        y_val_obs_combined,
        X_train_est_combined,
        X_val_est_combined,
        y_train_est_combined,
        y_val_est_combined,
    ) = prepare_data(
        train_observed_combined, train_estimated_combined, drop_features=drop_features
    )

    return (
        X_train_obs_combined,
        X_val_obs_combined,
        y_train_obs_combined,
        y_val_obs_combined,
        X_train_est_combined,
        X_val_est_combined,
        y_train_est_combined,
        y_val_est_combined,
    )


def get_preprocessed_test_data() -> pd.DataFrame:
    """
    Get the preprocessed test data without the 'date_forecast' column.
    """
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        X_test_estimated_a,
        X_test_estimated_b,
        X_test_estimated_c,
    ) = get_raw_data()

    # Align the test data to the same time as the training data
    X_test_estimated_a = temporal_alignment_tests(X_test_estimated_a)
    X_test_estimated_b = temporal_alignment_tests(X_test_estimated_b)
    X_test_estimated_c = temporal_alignment_tests(X_test_estimated_c)

    X_test_estimated_a = remove_missing_features(X_test_estimated_a)
    X_test_estimated_b = remove_missing_features(X_test_estimated_b)
    X_test_estimated_c = remove_missing_features(X_test_estimated_c)

    # Add location data
    X_test_estimated_a = add_location(X_test_estimated_a, "a")
    X_test_estimated_b = add_location(X_test_estimated_b, "b")
    X_test_estimated_c = add_location(X_test_estimated_c, "c")

    X_test_a_correct_features = feature_engineer(X_test_estimated_a)
    X_test_b_correct_features = feature_engineer(X_test_estimated_b)
    X_test_c_correct_features = feature_engineer(X_test_estimated_c)

    # X_train_obs_combined, X_val_obs_combined, y_train_obs_combined, y_val_obs_combined, X_train_est_combined, X_val_est_combined, y_train_est_combined, y_val_est_combined = fetch_preprocessed_data()

    # # Add historical data so that the model can use it for prediction
    # # Add mean_pv_measurement with same day and hour from previous years
    # X_test_estimated_a_with_historical_data = add_expected_pv_to_test_data(X_test_a_correct_features, X_train_obs_combined)
    # X_test_estimated_b_with_historical_data = add_expected_pv_to_test_data(X_test_b_correct_features, X_train_obs_combined)
    # X_test_estimated_c_with_historical_data = add_expected_pv_to_test_data(X_test_c_correct_features, X_train_obs_combined)

    # Drop the 'date_calc' and 'date_forecast' columns from the test data
    X_test_estimated_a_processed = X_test_a_correct_features.drop(
        columns=["date_calc", "date_forecast"], errors="ignore"
    )
    X_test_estimated_b_processed = X_test_b_correct_features.drop(
        columns=["date_calc", "date_forecast"], errors="ignore"
    )
    X_test_estimated_c_processed = X_test_c_correct_features.drop(
        columns=["date_calc", "date_forecast"], errors="ignore"
    )

    # # # Handle NaN values in the test data by filling them with the mean value of the respective column from the training data
    # X_test_estimated_a_processed.dropna()
    # X_test_estimated_b_processed.dropna()
    # X_test_estimated_c_processed.dropna()
    tests = pd.concat(
        [
            X_test_estimated_a_processed,
            X_test_estimated_b_processed,
            X_test_estimated_c_processed,
        ],
        ignore_index=True,
    )
    return tests


def interpolate_and_report(df, column="pv_measurement", max_gap=4):
    missing_idx = df[df[column].isnull()].index
    gaps = np.split(missing_idx, np.where(np.diff(missing_idx) != 1)[0] + 1)

    changes = []
    changes_count = 0

    for gap in gaps:
        if 1 <= len(gap) <= max_gap:
            start_idx = gap[0] - 1
            end_idx = gap[-1] + 1

            if start_idx < 0 or end_idx >= len(df):
                continue

            x_known = [start_idx, end_idx]
            y_known = df.loc[x_known, column].values

            interpolator = interp1d(
                x_known, y_known, kind="linear", fill_value="extrapolate"
            )

            x_missing = np.arange(start_idx + 1, end_idx)
            y_estimated = interpolator(x_missing)

            df.loc[x_missing, column] = y_estimated

            changes.append(
                {
                    "Time Before": df.loc[start_idx, "time"],
                    "Value Before": y_known[0],
                    "Time After": df.loc[end_idx, "time"],
                    "Value After": y_known[1],
                    "Interpolated Times": df.loc[x_missing, "time"].tolist(),
                    "Interpolated Values": y_estimated.tolist(),
                }
            )

            changes_count += len(gap)

    changes_df = pd.DataFrame(changes)

    print(f"Total changes made: {changes_count}")
    return df, changes_df
