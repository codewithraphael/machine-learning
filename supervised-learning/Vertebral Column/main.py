from src.data_loader import load_data
from src.preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model, roc_score, cross_validate
from src.predict import predict

import warnings
warnings.filterwarnings('ignore')

def main():
    vertebral_column = load_data('data/vertebral_column.csv')

    X_train, X_test, y_train, y_test, scaler = preprocess(vertebral_column)

    trained_model = train_model(X_train, y_train)

    # Evaluating each model in the dictionary
    for name, model in trained_model.items():
        print(f'\n ===== {name} Evaluation ===== ')
        y_score = evaluate_model(model, X_train, X_test, y_train, y_test)
        if y_score is not None:
            roc_score(y_test, y_score)

    # Cross-validate all trained models
    cross_validate(trained_model, X_train, y_train)

    # Sample Prediction
    sample = [63.0, 22.5, 39.6, 40.5, 98.3, 0.2]
    result = predict(trained_model, scaler, sample, model_key='xgb_model')

    print('\n Prediction:', result)


if __name__ == "__main__":
    main()