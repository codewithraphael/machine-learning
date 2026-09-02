from config import filepath, target_column, test_size
from data_loader import load_data
from eda import eda
from preprocessing import prepare_datetime, set_time_index
from plot import plot_time_series
from analysis import decompose_time_series, adf_test, diff_time_series
from forecasting import train_test_split


def main():

    data = load_data(filepath)
    eda(data)
    data = prepare_datetime(data, 'Month')
    data = set_time_index(data, 'Month')
    plot_time_series(data, 'Passengers')
    decomposition = decompose_time_series(data, 'Passengers', period=12)
    adf_test(data['Passengers'])
    diff_time_series(data, target_column)
    train, test = train_test_split(data, target_column, test_size)




if __name__ == '__main__':
    main()