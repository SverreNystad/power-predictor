import pandas as pd
import math

# Post processing:
# location A - 0 pv kl 23
# location C - 0 pv kl 22,,23,00
TIMES_TO_SET_0_PV_A = [23]
TIMES_TO_SET_0_PV_C = [22, 23, 0]
find_time_sin = lambda hour: math.sin(2 * math.pi * (hour) / 24)
find_time_cos = lambda hour: math.cos(2 * math.pi * (hour) / 24)




def set_0_pv_at_times(x_test: pd.DataFrame, y_pred: pd.DataFrame, location: str, hours: list[int]) -> pd.DataFrame:
    """Find the correct predicted values at the given times and locaiton and set them to 0."""
    hours_to_set_0_sin = [find_time_sin(hour) for hour in hours]
    hours_to_set_0_cos = [find_time_cos(hour) for hour in hours]


    indices = x_test[(x_test["location_" + location] == 1) & (x_test["sin_hour"].isin(hours_to_set_0_sin) & (x_test["cos_hour"].isin(hours_to_set_0_cos)))].index
    for index in indices:
        y_pred.loc[index] = 0
    return y_pred
