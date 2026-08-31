from config import filepath
from data_loader import load_data
from eda import eda
from preprocessing import prepare_datetime, set_time_index


def main():

    data = load_data(filepath)
    eda(data)
    data = prepare_datetime(data, 'Month')
    data = set_time_index(data, 'Month')



if __name__ == '__main__':
    main()