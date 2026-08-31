import pandas as pd

def prepare_datetime(data, date_column):

    data[date_column] = pd.to_datetime(data[date_column])

    return data


def set_time_index(data, date_column):

    data = data.set_index(data[date_column])

    print(f'\n ===== INDEXED DATE DATASET ===== \n {data.head(10)}')

    data = data.asfreq('MS')

    return data