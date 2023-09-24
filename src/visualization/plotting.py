from src.data.data_fetcher import get_all_features, get_raw_data
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

FIGURE_PATH = "results/figures/"


train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()

def parse_feature_name(feature_name: str) -> str:
    return feature_name.replace(':', '_')

def plot_single_feature(feature_name: str, show: bool = False) -> None:
    """
    Plots a single feature for all three train/test sets.
    """
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    X_train_observed_a[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[0], title='Train/Test A', color='red')
    X_train_estimated_a[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[0], title='Train/Test A', color='blue')
    X_test_estimated_a[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[0], title='Train/Test  A', color='green')

    X_train_observed_b[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[1], title='Train/Test  B', color='red')
    X_train_estimated_b[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[1], title='Train/Test  B', color='blue')
    X_test_estimated_b[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[1], title='Train/Test  B', color='green')

    X_train_observed_c[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[2], title='Train/Test  C', color='red')
    X_train_estimated_c[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[2], title='Train/Test  C', color='blue')
    X_test_estimated_c[['date_forecast', feature_name]].set_index('date_forecast').plot(ax=axs[2], title='Train/Test  C', color='green')

    file_name = parse_feature_name(feature_name)

    plt.ylabel(feature_name)
    plt.xlabel('Date')
    plt.savefig(f'{FIGURE_PATH}time_plot/' + file_name + '.png')
    plt.legend(['Observed', 'Estimated', 'Test'])
    if show:
        plt.show()
    plt.close()

def plot_all_features() -> None:
    """
    Plots all features for all three train/test sets.
    """
    for feature in get_all_features():
        print(f"[INFO] Plotting {feature}")
        plot_single_feature(str(feature))

def scatter_plot(feature_name: str, show: bool = False) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

    X_train_observed_a[['date_forecast', feature_name]].set_index('date_forecast').plot(title='Train/Test A', kind='scatter', x='date_forecast', y=feature_name, color='red')
    X_train_estimated_a[['date_forecast', feature_name]].set_index('date_forecast').plot(title='Train/Test A', kind='scatter', x='date_forecast', y=feature_name, color='blue')
    X_test_estimated_a[['date_forecast', feature_name]].set_index('date_forecast').plot(title='Train/Test A', kind='scatter', x='date_forecast', y=feature_name, color='green')

    X_train_observed_b[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='red')
    X_train_estimated_b[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='blue')
    X_test_estimated_b[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='green')

    X_train_observed_c[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='red')
    X_train_estimated_c[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='blue')
    X_test_estimated_c[['date_forecast', feature_name]].set_index('date_forecast').plot(kind='scatter', x='date_forecast', y=feature_name, color='green')
    file_name = parse_feature_name(feature_name)
    plt.ylabel(feature_name)
    plt.xlabel('Date')
    plt.savefig(f'{FIGURE_PATH}scatter_plot/' + file_name + '.png')
    if show:
        plt.show()
    
    plt.close()

def box_plot(feature_name: str, show: bool = False) -> None:
    # One way we can extend this plot is adding a layer of individual points on top of
    # it through Seaborn's striplot
    # 
    # We'll use jitter=True so that all the points don't fall in single vertical lines
    # above the species
    #
    # Saving the resulting axes as ax each time causes the resulting plot to be shown
    # on top of the previous axes
    # ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
    # ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
    # ax.

    pass

def pair_grid_plot(feature_name: str, show: bool = False) -> None:
    # We can quickly make a boxplot with Pandas on each feature split out by species
    # iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

    sns.pairplot(X_train_observed_a.drop("date_forecast", axis=1), hue="date_forecast", height=10)
    plt.savefig(f'{FIGURE_PATH}pair_grid_plot/' + feature_name + '.png')

if __name__ == '__main__':
    X_train_observed_a: pd.DataFrame
    print(X_train_observed_a.keys())
    features_left = len(X_train_observed_a.keys())
    for feature in X_train_observed_a.keys():
        print(f"[INFO] Plotting {feature}, {features_left} left")
        features_left -= 1
        if feature == 'date_forecast':
            continue
        scatter_plot(str(feature))

def plot_moving_average(series, window, plot_intervals=False, scale=1.96, title="Moving Average Trend"):
    """
    Compute and plot moving average for given series. 
    For the moving average, we take the average of the past few days of the time series.
    This is good for smoothing out short-term flucuations and highlighting long-term trends.
    And trend analysis is what we are after.

    Args: 
        series - dataset with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        scale - scaling factor for confidence intervals
    """
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(15,5))
    plt.title(title)
    plt.plot(rolling_mean, 'g', label='Rolling Mean Trend')
    
    # Plot confidence intervals for the moving average
    if plot_intervals:
        mae = series.rolling(window=window).std()
        deviation = mae * scale
        lower_bound = rolling_mean - deviation
        upper_bound = rolling_mean + deviation
        plt.fill_between(x=series.index, y1=lower_bound, y2=upper_bound, color='b', alpha=0.2)
    
    plt.plot(series, label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def detect_outliers(target_series, title: str):
    """
    Detects outliers in a time series using the IQR method and plots the time series with outliers highlighted.
    
    Args:
    - target_series: A pandas Series representing the time series data.
    
    Returns:
    - A plot of the time series with outliers highlighted in red.
    """
    
    # IQR method for outlier detection
    Q1 = target_series.quantile(0.25)
    Q3 = target_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = target_series[(target_series < lower_bound) | (target_series > upper_bound)]

    # Plotting the time series and outliers
    plt.figure(figsize=(15,6))
    plt.plot(target_series.index, target_series, label='Target Values', color='blue')
    plt.scatter(outliers.index, outliers, color='red', label='Outliers')
    plt.title(title)
    plt.legend()
    plt.show()


def seasonal_trends(targets: pd.DataFrame, title: str, show: bool = False):
    """
    Decomposes a time series into its trend, seasonal, and residual components.
    
    Args:
        targets: A pandas DataFrame representing the time series data.
        title: A string representing the title of the plot.
        show: A boolean indicating whether or not to display the plot.
    """

    result = seasonal_decompose(targets['pv_measurement'], model='additive', period=24)
    result.plot()
    #Save
    plt.savefig(f'{FIGURE_PATH}seasonal_trends/' + title + '.png')
    if show:
        plt.show()
    plt.close()
    # Seasonal plot for daily patterns
    daily_seasonal = result.seasonal['2022-01-01':'2022-01-02']  # Adjust dates to pick a representative 2-day period
    daily_seasonal.plot(figsize=(15,6))
    plt.title('Daily Seasonal Pattern')
    plt.savefig(f'{FIGURE_PATH}seasonal_trends/' + title + 'from_2022-01-01_to_2022-01-02.png')
    if show:
        plt.show()
    plt.close()

    # Autocorrelation plot to identify seasonality
    plot_acf(targets['pv_measurement'], lags=168)  # 168 hours for a weekly pattern
    plt.title('Autocorrelation Plot')
    plt.savefig(f'{FIGURE_PATH}seasonal_trends/' + title + 'autocorrelation.png')
    if show:
        plt.show()
    plt.close()