from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from config import PLOTS_PATH


def decompose_time_series(data, target_column, period):

    plt.figure(figsize=(16, 8))

    decomposition = seasonal_decompose(
        data[target_column],
        model='multiplicative',
        period=period
    )

    decomposition.plot()
    
    plt.savefig(PLOTS_PATH / 'decomposition_plot.png')
    plt.close()

    return decomposition


def adf_test(series):

    result = adfuller(series.dropna())

    statistic = result[0]
    p_value = result[1]

    print('='*50)
    print('                ' + 'ADF Statistics')
    print('='*50)
    print(f'\n{statistic:.4f}')


    print('='*50)
    print('                 ' + 'P - Value')
    print('='*50)
    print(f'\n{p_value:.4f}')

    if p_value < 0.05:
        print('The series is likely stationary')
    else:
        print('The series is likely non stationary')