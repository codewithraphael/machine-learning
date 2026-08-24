import numpy as np
from config import TARGET, HORIZON, SENSOR_FEATURES, FUTURE_TARGET

def prepare_temporal_order(data):

    '''
    observations are sorted in temporal order, 
    so that the model can learn from the past to predict the future.
    '''

    data = data.copy()
    data = data.sort_values('timestamp').reset_index(drop=True)

    return data

def create_time_index(data):

    '''
    add a sequential integer representing the chronological position of each observation
    '''

    data = data.copy()
    data['time_index'] = np.arange(len(data))

    return data

def create_future_target(data, target=TARGET, horizon=HORIZON):
 
    '''
    creating a future target for time series forecasting.

    The target at time t is shifted backwards so that features at time t
    are used to predict the target at t + horizon
    '''

    if horizon < 1:
        raise ValueError('horizon must be a positive integer')

    if target not in data.columns:
        raise KeyError(f"target column '{target}' not found in data")

    data = data.copy()

    future_target = FUTURE_TARGET

    data[future_target] = (
        data[target].shift(-horizon)
    )

    data = data.dropna(subset=[future_target])

    data[future_target] = (
        data[future_target].astype(int)
    )

    print(data[future_target].value_counts())

    return data


def create_lagged_features(data, feature_columns=SENSOR_FEATURES, lags=(1, 2, 3, 5, 10)):

    '''
    create lagged features for time series forecasting.

    For each feature in feature_columns, create new columns representing the values of that feature
    at previous time steps (lags). This allows the model to learn from past observations to predict future outcomes.
    '''

    data = data.copy()

    for feature in feature_columns:
        for lag in lags:
            lagged_feature = f'{feature}_lag_{lag}'
            data[lagged_feature] = data[feature].shift(lag)

    data = data.dropna()

    return data

def create_rolling_features(data, feature_columns=SENSOR_FEATURES, windows=(3, 5, 10)):

    '''
    create rolling mean, standard deviation, minimum and maximum features from historical sensor observation.

    '''

    data = data.copy()

    for column in feature_columns:

        if column in data.columns:
            continue

        for window in windows:

            shifted = data[column].shift(1)

            data[f'{column}_rolling_mean{window}'] = (
                shifted.rolling(window).mean()
            )
            data[f'{column}_rolling_std{window}'] = (
                shifted.rolling(window).std()
            )
            data[f'{column}_rolling_min{window}'] = (
                shifted.rolling(window).min()
            )
            data[f'{column}_rolling_max{window}'] = (
                shifted.rolling(window).max()
            )

    return data


def remove_invalid_rows(data):

    '''
    remove rows that contain insufficient historical observations
    after lag and rolling features creation.
    '''

    data = data.copy()

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.dropna(
        how='all'
    )

    return data.reset_index(drop=True)