from src.data.data_fetcher import get_all_features, get_raw_data
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pywt


FIGURE_PATH = "results/figures/"


train_a, train_b, train_c, X_train_estimated_a, X_train_estimated_b, X_train_estimated_c, X_train_observed_a, X_train_observed_b, X_train_observed_c, X_test_estimated_a, X_test_estimated_b, X_test_estimated_c = get_raw_data()

def parse_feature_name(feature_name: str) -> str:
    return feature_name.replace(':', '_')

def plot_pv_measurement(show: bool = False) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    train_a.set_index('time').plot(ax=axs[0], title='location A', color='red')
    train_b.set_index('time').plot(ax=axs[1], title='location B', color='red')
    train_c.set_index('time').plot(ax=axs[2], title='location C', color='red')


    plt.ylabel('PV Measurement')
    plt.xlabel('Date')
    plt.savefig(f'{FIGURE_PATH}time_plot/pv_measurement.png')
    if show:
        plt.show()
    plt.close()

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


def plot_boxplots(df: pd.DataFrame, title: str, show: bool = False) -> None:
    """
    Plot box plots for each feature in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the features.
    """
    # Filter out non-numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Number of numeric features
    num_features = df_numeric.shape[1]  
    
    if num_features == 0:
        print("No numeric columns to plot")
        return
    
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 4 * num_features))
    
    # Check if df has only one numeric feature, if so, axes is not an array and needs to be put into one
    if num_features == 1:
        axes = [axes]
    
    for i, col in enumerate(df_numeric.columns):
        axes[i].boxplot(df_numeric[col].dropna(), vert=True)
        axes[i].set_title(f'Box plot of {col}')
        axes[i].set_ylabel('Values')
        
    plt.tight_layout()
    plt.savefig(f'{FIGURE_PATH}boxplots/{title}_boxplots.png')
    if show:
        plt.show()

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
    plt.plot(rolling_mean, 'g', label='Rolling Mean Trend')
    plt.savefig(f'{FIGURE_PATH}moving_average/' + title + '.png')

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


def spectral_analysis(signal: pd.Series, title: str, show: bool = False) -> None:
    """
    Spectral analysis of a time series using Fourier Transform.

    Args:
        signal: A pandas Series representing the time series data.
        title: A string representing the title of the plot.
        show: A boolean indicating whether or not to display the plot.
    """

    # Apply Fourier Transform using numpy
    spectral_density = np.abs(np.fft.fft(signal))

    # Frequency values for the x-axis
    frequencies = np.fft.fftfreq(len(spectral_density))

    # Plot Spectral Density
    plt.figure(figsize=(10,6))
    plt.plot(frequencies, spectral_density)
    plt.title('Spectral Density')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.savefig(f'{FIGURE_PATH}spectral_analysis/spectral_density_{title}.png')    
    if show:
        plt.show()
    plt.show()

def wavelet_analysis(signal: pd.Series, title: str, wavelet: str = "cmor", show: bool = False) -> None:
    """
    Wavelet analysis of a time series using Continuous Wavelet Transform.
    Continuous Wavelet Transform is a mathematical tool used to analyze non-stationary signals.
        * Morlet Wavelet ('cmor'): Often used for analyzing oscillatory patterns and is suitable for most applications due to its good frequency localization.
        * Mexican Hat Wavelet ('mexh'): Suitable for detecting sharp changes in signals.

    """
    # Perform Continuous Wavelet Transform
    coefficients, frequencies = pywt.cwt(signal, scales=np.arange(1, 128), wavelet=wavelet)

    # Plot Wavelet Transform result
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, len(signal), 1, 128], cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.savefig(f'{FIGURE_PATH}wavelet_analysis/wavelet_{wavelet}_transform_{title}.png')
    if show:
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame, title: str, show: bool = False) -> None:
    """
    Plots a heatmap of the correlation between the features in the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        title (str): The title of the plot.
        show (bool): A boolean indicating whether or not to display the plot.
    """
    correlation_matrix = df.corr()
    # Compute the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig(f'{FIGURE_PATH}feature_correlation/{title}_correlation.png')
    if show:
        plt.show()

def plot_acf_daily_weekly_monthly_yearly(date_frame: pd.DataFrame, feature: str, title: str, show: bool = False) -> None:
    """
    Plot the Autocorrelation Function (ACF) for a given feature in a DataFrame.
    The ACF is a measure of the correlation between the time series and a lagged version of itself.
    This is useful for identifying patterns in the time series data.
    
    """

    plt.figure(figsize=(15,6))
    plot_acf(date_frame[feature], lags=24)  # 24 hours to check for daily patterns
    plt.title(f'Autocorrelation Function (ACF) Plot for {title} (Daily)')
    plt.savefig(f"{FIGURE_PATH}acf/{feature}_daily_for_{title}.png")
    if show:
        plt.show() 

    # Plot the Autocorrelation Function
    plt.figure(figsize=(15,6))
    plot_acf(date_frame[feature], lags=168)  # 168 hours to check for weekly patterns
    plt.title(f'Autocorrelation Function (ACF) Plot for {title} (Weekly)')
    plt.savefig(f"{FIGURE_PATH}acf/{feature}_weekly_for_{title}.png")
    if show:
        plt.show()


    plt.figure(figsize=(15,6))
    plot_acf(date_frame[feature], lags=168*4)  # 168*4 hours to check for monthly patterns
    plt.title(f'Autocorrelation Function (ACF) Plot for {title} (Monthly)')
    plt.savefig(f"{FIGURE_PATH}acf/{feature}_monthly_for_{title}.png")
    if show:
        plt.show()

    plt.figure(figsize=(15,6))
    plot_acf(date_frame[feature], lags=168*4*12)  # 168*4*12 hours to check for yearly patterns
    plt.title(f'Autocorrelation Function (ACF) Plot for {title} (Yearly)')
    plt.savefig(f"{FIGURE_PATH}acf/{feature}_yearly_for_{title}.png")
    if show:
        plt.show()