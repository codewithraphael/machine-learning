import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import PLOTS_PATH


def plot_target_series(series):

    plt.figure(figsize=(20, 8))

    plt.plot(series)

    plt.title('Historical Closing Price')
    plt.xlabel('Date')
    plt.ylabel('CLosing Price')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOTS_PATH / 'target_distribution.png')
    plt.close()


def plot_rolling_statistics(series, window=30):

    '''
    plot the original series together with its rolling mean and standard deviation.
    '''

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()

    fig, ax = plt.subplots(figsize=(24, 8))

    ax.plot(series, label='Close')
    ax.plot(rolling_mean, label = f'{window} Days Rolling Mean')
    ax.plot(rolling_std, label = f'{window} Days Rolling Standard Deviation')
    ax.plot(rolling_min, label = f'{window} Days Rolling Minumum')
    ax.plot(rolling_max, label = f'{window} Days Rolling Maximum')
    ax.set_title('Rolling Statistics')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.savefig(PLOTS_PATH / 'rolling_statistics.png')
    plt.close()

def plot_acf_pacf(series, lags=40):

    '''
    plot autocorrelation and partial autocorrelation function for the time series
    '''

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm')

    axes[0].set_title('Autocorrelation Function Plot')
    axes[1].set_title('Partial AUtocorrelation Function Plot')

    fig.tight_layout()

    plt.savefig(PLOTS_PATH / 'acf_pacf_plots.png')
    plt.close()