from config import file_path, BINARY_COLUMNS, TARGET, HORIZON, SENSOR_FEATURES
from data_loader import load_data
from data_cleaning import clean_data, investigate_time
from feature_engineering import prepare_temporal_order, create_time_index, create_future_target, create_lagged_features
from plots import plot_grip_loss_over_time

'''
from train import train_model
from evaluate import evaluate_model
'''

import warnings; warnings.filterwarnings('ignore')


def main():
    data = load_data(file_path)
    data = clean_data(data, BINARY_COLUMNS)
    investigate_time(data)
    data = prepare_temporal_order(data)
    data = create_time_index(data)
    plot_grip_loss_over_time(data)
    data = create_future_target(data, target=TARGET, horizon=HORIZON)
    data = create_lagged_features(data, feature_columns=SENSOR_FEATURES, lags=(1, 2, 3, 5, 10))



if __name__ == '__main__':
    main()