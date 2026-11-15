from config import DATA_DIR, MODELS_DIR, PLOTS_DIR, EVAL_RESULT_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE, FEATURE_COLS
from data_loader import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from data_visualization import visualize_data
from preprocessing import split_data, preprocess_data
from train import train_models
from evaluation import evaluate_model


import warnings
warnings.filterwarnings('ignore')


def main():

    data = load_data(DATA_DIR)
    validate_data(data)
    clean_data(data)
    visualize_data(data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, test_size=TEST_SIZE, validation_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, X_test, preprocessor = preprocess_data(X_train, X_val, X_test)

    trained_models = train_models(X_train, y_train)

    for name, model in trained_models.items():
        evaluate_model(X_test, X_val, y_test, y_val, model, name, features_names=FEATURE_COLS)

if __name__ == '__main__':
    main()