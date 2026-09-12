import pandas as pd
import numpy as np

def create_returns(data, price_columns='close'):

    data = data.copy()

    data['returns'] = data[price_columns].pct_change()

    print('='*100)
    print(' '*30 + 'STOCK PRICE DATA WITH RETURNS VALUES')
    print('='*100)

    print(data.head(5))
    print(data.tail(5))

    return data


def create_log_returns(data, price_columns='close'):

    data = data.copy()

    data['log_returns'] = np.log(data[price_columns] / data[price_columns].shift(1))

    print('='*100)
    print(' '*30 + 'STOCK PRICE DATA WITH LOG RETURNS VALUES')
    print('='*100)

    print(data.head(5))
    print(data.tail(5))

    return data


def create_lag_features(data, column='close', lags=5):

    data = data.copy()

    for lag in range(1, lags + 1):
        data[f'{column}_lag_{lag}'] = data[column].shift(lag)

    print('='*100)
    print(' '*30 + 'STOCK PRICE DATA WITH LAG FEATURES')
    print('='*100)

    print(data.head(5))
    print(data.tail(5))

    return data



def create_rolling_features(data, column='close', windows=(7, 30, 60)):

    data = data.copy()

    for window in windows:

        data[f'{column}_rolling_mean_{window}'] = (
            data[column].rolling(window).mean()
        )

        data[f'{column}_rolling_std_{window}'] = (
            data[column].rolling(window).std()
        )

    print('='*100)
    print(' '*30 + 'STOCK PRICE DATA WITH ROLLING WINDOW FEATURES')
    print('='*100)

    print(data.head(5))
    print(data.tail(5))
    
    return data