from src.data_loader import load_data
from src.eda import eda, outlier_detection, outlier_check, plot_outliers, plot_pairplot, plot_correlation_matrix


def main():
    filepath = 'data/synthetic_heart_attack_dataset.csv'
    data = load_data(filepath)
    data = data.drop(columns=['patient_id'], axis=1)
    
    eda(data)

    lower_bound, upper_bound = outlier_detection(data)
    outlier_check(data, lower_bound, upper_bound)

    plot_outliers(data)
    plot_pairplot(data)
    plot_correlation_matrix(data)


if __name__ == "__main__":
    main()