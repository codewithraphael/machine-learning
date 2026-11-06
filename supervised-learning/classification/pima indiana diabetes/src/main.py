from config import DATA_DIR, FEARTURE_COLUMNS, MODEL_DIR, PLOTS_DIR, EVAL_RESULT_DIR
from data_loader import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from visualization import visualize_data
from preprocessing import feature_selection, preprocess_data
from train import train_models
from evaluation import evaluate_model




def main():

    data = load_data(DATA_DIR)
    validate_data(data)
    clean_data(data)
    visualize_data(data)
    X_train, X_test, y_train, y_test = feature_selection(data)
    preprocessor, X_train, X_test = preprocess_data(X_train, X_test)
    trained_models = train_models(X_train, y_train)

    for name, model in trained_models.items():
        evaluate_model(X_train, X_test, y_train, y_test, model, FEARTURE_COLUMNS)



if __name__ == '__main__':
    main()