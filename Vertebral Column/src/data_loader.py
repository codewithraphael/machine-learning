import pandas as pd

def load_data(filepath):
    '''
    Loading data and returning output

    '''

    data = pd.read_csv(filepath)

    return data