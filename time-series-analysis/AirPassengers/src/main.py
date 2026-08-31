from config import filepath
from data_loader import load_data
from eda import eda
from preprocessing import prepare_datetime, set_time_index
from plot import plot_time_series


def main():

    data = load_data(filepath)
    eda(data)
    data = prepare_datetime(data, 'Month')
    data = set_time_index(data, 'Month')
    plot_time_series(data, 'Passengers')



if __name__ == '__main__':
    main()