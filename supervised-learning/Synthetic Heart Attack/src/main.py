from data_loader import load_data
from config import drop_column
from eda import eda, outlier_detection, outlier_check, plot_outliers, plot_pairplot, plot_correlation_matrix
from preprocessing import feature_selection, preprocess_data
from train import train_models
from eval import evaluate, plot_confusion_matrices, plot_roc_curves
from eval import save_best_model

import warnings
warnings.filterwarnings('ignore')


def main():
    filepath = '../data/synthetic_heart_attack_dataset.csv'
    data = load_data(filepath)
    data = data.drop(drop_column, axis=1)
    
    eda(data)
    lower_bound, upper_bound = outlier_detection(data)
    outlier_check(data, lower_bound, upper_bound)

    plot_outliers(data)
    plot_pairplot(data)
    plot_correlation_matrix(data)

    X, y, X_train, X_test, y_train, y_test = feature_selection(data)
    preprocessor = preprocess_data(X)
    
    name, trained_models, pipe = train_models(preprocessor, X_train, y_train)

    for name, pipe in trained_models.items():
        y_pred, y_prob = evaluate(name, pipe, X_train, X_test, y_train, y_test)
    
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)

    save_best_model(trained_models)



if __name__ == "__main__":
    main()