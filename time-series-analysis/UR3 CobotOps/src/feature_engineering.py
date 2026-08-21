def prepare_temporal_order(data):

    '''
    observations are sorted in temporal order, 
    so that the model can learn from the past to predict the future.
    '''

    data = data.copy()
    data = data.sort_values('timestamp').reset_index(drop=True)

    return data