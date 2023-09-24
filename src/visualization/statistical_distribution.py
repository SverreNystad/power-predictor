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

def anderson_darling_test(target_series):
    """
    Analyzes the normality of a time series using the Anderson-Darling test.
    
    Parameters:
    - target_series: A pandas Series representing the time series data.
    
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

    # Assume 'data' is the variable containing your sample data
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