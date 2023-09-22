from src.data.data_fetcher import get_all_features, get_raw_data
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

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

    