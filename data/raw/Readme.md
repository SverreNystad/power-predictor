# Solar Dayahead Forecast Data

This dataset provides data for evaluating solar production dayahead forecasting methods.
The dataset contains three locations (A, B, C), corresponding to office buildings with solar panels installed.
There is one folder for each location.

There are 4 files in each folder:

1. train_targets.parquet - target values for the train period (solar energy production)
2. X_train_observed.parquet - actual weather features for the first part of the training period
2. X_train_estimated.parquet - predicted weather features for the remaining part of the training period
2. X_test_estimated.parquet - predicted weather features for the test period

For Kaggle submissions we have two more files: 
1. test.csv — test file with zero predictions (for all three locations)
2. sample_submission_kaggle.csv — sample submission in the Kaggle format (for all three locations)

Kaggle expects you to submit your solutions in the "sample_sumbission_kaggle.csv" format. Namely, you need to have two columns: "id" and "prediction".
The correspondence between id and time/location is in the test.csv file. An example solution is provided in "read_files.ipynb"

All files that are in the parquet format that can be read with pandas:
```shell
pd.read_parquet()
```

Baseline and targets production values have hourly resolution.
Weather has 15 min resolution.
Weather parameter descriptions can be found [here](https://www.meteomatics.com/en/api/available-parameters/alphabetic-list/).

There is a distinction between train and test weather data.
For training we have both observed weather data and its forecasts, while for test we only have forecasts.
While file `train_weather.parquet` contains one time column `date_forecast` to indicate when the values for the current row apply,
`test_weather.parquet` also contains `date_calc` to indicate when the forecast was produced.
This type of test data makes evaluation closer to how the forecasting methods that are used in production.

Evaluation measure is [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error).
