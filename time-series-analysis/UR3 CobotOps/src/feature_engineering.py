import numpy as np
from config import TARGET, HORIZON

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

    future_target = f'{target}_t_plus_{horizon}'

    data[future_target] = (
        data[target].shift(-horizon)
    )

    data = data.dropna(subset=[future_target])

    data[future_target] = (
        data[future_target].astype(int)
    )

    return data