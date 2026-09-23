from config import DATA_PATH, MODELS_PATH, PLOTS_PATH, RESULTS_PATH, CAT_COLUMNS, NUM_COLUMNS
from data_loader import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from preprocessing import feature_selection, preprocess_data
from train import train_model
from evaluation import evaluate_model, plot_model_comparison

import warnings
warnings.filterwarnings("ignore")



def main():

    data = load_data(DATA_PATH)
    validate_data(data)
    data = clean_data(data)
    X_train, X_test, y_train, y_test, label_encoder = feature_selection(data)
    transformer, X_train, X_test = preprocess_data(X_train, X_test)
    trained_models = train_model(X_train, y_train)

    for name, model in trained_models.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test)

    plot_model_comparison(trained_models, X_test, y_test)




if __name__ == '__main__':
    main()