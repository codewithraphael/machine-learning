from config import DATA_DIR, PLOTS_DIR, EVAL_RESULT_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE, FEATURE_COLS
from data_loader import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from data_visualization import visualize_data
from preprocessing import split_data
from train import train_models
from evaluation import evaluate_model, plot_roc_curves, plot_confusion_matrices, plot_validation_curves
from utils import save_model


import warnings
warnings.filterwarnings('ignore')


def main():

    data = load_data(DATA_DIR)
    validate_data(data)
    clean_data(data)
    visualize_data(data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, test_size=TEST_SIZE, validation_size=VAL_SIZE, random_state=RANDOM_STATE)
    trained_models = train_models(X_train, y_train)

    for name, model in trained_models.items():
        evaluate_model(X_test, X_val, y_test, y_val, model, name, features_names=FEATURE_COLS)

    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_validation_curves(trained_models, X_train, y_train)

    save_model(trained_models['XGBOOST CLASSIFIER'])
if __name__ == '__main__':
    main()