import numpy as np

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

    