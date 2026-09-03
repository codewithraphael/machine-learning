import pandas as pd

from config  import target_column, test_size

def train_test_split(data, target_column, test_size):

    series = data[target_column]

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    print('Training Observation: ', len(train))
    print('Testing Observation: ', len(test))

    return train, test


def naive_forecast(train, test):

    baseline_predictions = pd.Series(train.iloc[-1], index=test.index)


    return baseline_predictions