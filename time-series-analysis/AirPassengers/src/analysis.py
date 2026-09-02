from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import target_column

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from config import PLOTS_PATH


def decompose_time_series(data, target_column, period):

    plt.Figure(figsize=(20, 8))

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
    print(' '*16 + 'ADF Statistics')
    print('='*50)
    print(f'\n{statistic:.4f}')


    print('='*50)
    print(' '*17 + 'P - Value')
    print('='*50)
    print(f'\n{p_value:.4f}')

    if p_value < 0.05:
        print('The series is likely stationary')
    else:
        print('The series is likely non stationary')



def diff_time_series(data, target_column):

    print('='*90)
    print(' '*30 + 'DIFFERENCING')
    print('='*90)

    data['Passengers_diff'] = data[target_column].diff()
    adf_test(data['Passengers_diff'])

    data['Passengers_diff'].plot(figsize=(15, 5))
    plt.title('Differenced Passenger Series')
    plt.xlabel('Date')
    plt.ylabel('Difference')

    plt.savefig(PLOTS_PATH / 'passenger_differencing.png')
    plt.close()