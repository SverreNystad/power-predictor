import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

def analyze_normality(target_series):
    """
    Analyzes the normality of a time series using the Shapiro-Wilk test.
    
    Parameters:
    - target_series: A pandas Series representing the time series data.
    
    Returns:
    - Result of the Shapiro-Wilk test and insights on the distribution.
    """
    
    # Perform Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(target_series)
    
    # Check for normality based on the p-value
    alpha = 0.05
    if shapiro_p > alpha:
        print("Shapiro-Wilk Test: Data seems to be normally distributed (fail to reject H0).")
    else:
        print("Shapiro-Wilk Test: Data does not seem to be normally distributed (reject H0).")
    
    # Return test statistic and p-value
    return shapiro_stat, shapiro_p

def anderson_darling_test(target_series: pd.Series):
    """
    Analyzes the normality of a time series using the Anderson-Darling test.
    
    Args:
        target_series (pandas.Series): A pandas Series representing the time series data.
    
    Returns:
    - Result of the Anderson-Darling test and insights on the distribution.
    """
    
    # Perform Anderson-Darling test
    anderson_stat, anderson_crit_vals, anderson_sig_levels = stats.anderson(target_series)
    
    # Check for normality based on the critical values
    alpha = 0.05
    if anderson_stat < anderson_crit_vals[2]:
        print("Anderson-Darling Test: Data seems to be normally distributed (fail to reject H0).")
    else:
        print("Anderson-Darling Test: Data does not seem to be normally distributed (reject H0).")
    
    # Return test statistic and critical values
    return anderson_stat, anderson_crit_vals

def lilliefors_test(target_series):
    """
    Analyzes the normality of a time series using the Lilliefors test.
    It is better for large samples.

    Args:
        target_series (pandas.Series): A pandas Series representing the time series data.
    Returns: 
        Result of the Lilliefors test and insights on the distribution.
    """
    
    result = sm.stats.diagnostic.lilliefors(target_series, dist='norm')

    statistic, p_value = result
    print(f"Test Statistic: {statistic}")
    print(f"P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Data does not look normal (Reject H0)")
    else:
        print("Data looks normal (Fail to reject H0)")

        return statistic, p_value
    

def feature_correlation(df: pd.DataFrame, threshold: float = 0.8):
    """
    Finds pairs of features with correlation above a given threshold.

    Args:
        df (pandas.DataFrame): A pandas DataFrame representing the data.
        threshold (float, optional): The threshold for the correlation. Defaults to 0.8.
    """
    correlation_matrix = df.corr()

    # Find pairs of features with correlation above threshold
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    
    return(correlated_features)