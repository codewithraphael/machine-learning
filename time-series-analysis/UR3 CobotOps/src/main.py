from config import file_path, BINARY_COLUMNS, TARGET, HORIZON, SENSOR_FEATURES, FUTURE_TARGET, TRAIN_RATIO
from data_loader import load_data
from data_cleaning import clean_data, investigate_time
from feature_engineering import prepare_temporal_order, create_time_index, create_future_target, create_lagged_features, create_rolling_features, remove_invalid_rows
from plots import plot_grip_loss_over_time
from preprocessing import feature_selection, split_data, preprocess_data
from train import train_models
from evaluation import evaluate_models

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
    data = create_rolling_features(data, feature_columns=SENSOR_FEATURES, windows=(3, 5, 10))
    data = remove_invalid_rows(data)
    features = feature_selection(data, sensor_features=SENSOR_FEATURES, future_target=FUTURE_TARGET)
    X_train, X_test, y_train, y_test = split_data(data, features, target=FUTURE_TARGET, train_ratio=TRAIN_RATIO)
    preprocessor = preprocess_data()
    trained_models, pipe = train_models(preprocessor, X_train, y_train)

    for name, pipe in trained_models.items():
        y_pred, y_prob = evaluate_models(name, pipe, X_train, X_test, y_train, y_test)






if __name__ == '__main__':
    main()