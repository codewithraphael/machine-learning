from config import (
    BASE_DIR, 
    DATA_DIR, 
    MODEL_DIR, 
    PLOTS_DIR, 
    EVAL_RESULT_DIR, 
    TARGET_COLUMN, 
    REQUIRED_COLUMNS, 
    TEXT_COLUMN, 
    TEST_SIZE, 
    RANDOM_STATE
)
from data_loader import load_data
from data_validation import validate_data

from eda import (eda, 
    visualize_target_distribution
)

from preprocessing import build_classifier

from train import (
    split_features,
    train_model
)

from evaluation import evaluate_model
from utils import save_model

import warnings; warnings.filterwarnings('ignore')


def main():
    data = load_data(DATA_DIR)
    validate_data(data)
    eda(data, TEXT_COLUMN)
    visualize_target_distribution(data, TARGET_COLUMN, PLOTS_DIR)

    trained_models = {}

    for target in TARGET_COLUMN:
        x_train, x_test, y_train, y_test = split_features(data, target)
        pipe = build_classifier()
        pipe, best_params = train_model(x_train, y_train, pipe, tune=True)
        print(f'{target}: fitted')
        if best_params:
            print(f'{target}: best params {best_params}')

        evaluate_model(pipe, x_train, x_test, y_train, y_test, target_name=target)
        trained_models[target] = pipe

    save_model(trained_models)

    


if __name__ == '__main__':
    main()