import pandas as pd

from config import filepath, target_column, test_size, order, seasonal_order
from data_loader import load_data
from eda import eda
from preprocessing import prepare_datetime, set_time_index
from plot import plot_time_series
from analysis import decompose_time_series, adf_test, diff_time_series
from forecasting import train_test_split, naive_forecast
from evaluation import evaluate_forecast, fit_arima, generate_forecast, plot_forecast, fit_sarima, model_diagnostics, acf_plot, pacf_plot
from utils import save_model


def main():

    data = load_data(filepath)
    eda(data)
    data = prepare_datetime(data, 'Month')
    data = set_time_index(data, 'Month')
    plot_time_series(data, 'Passengers')
    decomposition = decompose_time_series(data, 'Passengers', period=12)
    adf_test(data['Passengers'])
    diff_time_series(data, target_column)
    train, test = train_test_split(data, target_column, test_size)
    baseline_predictions = naive_forecast(train, test)
    baseline_metrics = evaluate_forecast(test, baseline_predictions)

    arima_model = fit_arima(train, order=order)
    arima_predictions = generate_forecast(arima_model, steps = len(test))
    arima_metrics = evaluate_forecast(test, arima_predictions)
    print(arima_metrics)

    plot_forecast(test, arima_predictions)

    sarima_model = fit_sarima(train, order=order, seasonal_order=seasonal_order)
    sarima_prediction = generate_forecast(sarima_model, steps=len(test))
    sarima_metrics = evaluate_forecast(test, sarima_prediction)
    print(sarima_metrics )

    results = pd.DataFrame({
        'Baseline Metrics': baseline_metrics,
        'ARIMA Metrics': arima_metrics,
        'SARIMA Metrics': sarima_metrics
    })

    print('='*80)
    print(' '*25 + 'MODELS METRICS COMPARISON')
    print('='*80)

    print(results)

    model_diagnostics(model=sarima_model)
    acf_plot(data, target_column)
    pacf_plot(data, target_column)

    save_model(sarima_model)



if __name__ == '__main__':
    main()