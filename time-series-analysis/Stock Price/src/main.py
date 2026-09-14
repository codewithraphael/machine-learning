from config import TARGET_COLUMN, TEST_SIZE
from data_loader import load_data
from eda import eda, visualize_data
from preprocessing import clean_data, select_target, split_time_series
from analysis_plots import plot_target_series, plot_rolling_statistics, plot_acf_pacf
from feature_engineering import create_returns, create_log_returns, create_lag_features, create_rolling_features
from forecasting import naive_forecasting, fit_arima, forecast_arima, fit_sarimax, forecast_sarimax, fit_exponential_smoothing, forecast_exponential_smoothing, plot_forecast_model

import warnings
warnings.filterwarnings('ignore')




def main():

    data = load_data()
    eda(data)
    visualize_data(data)
    data = clean_data(data)
    series = select_target(data, TARGET_COLUMN)
    train, test = split_time_series(series, TEST_SIZE)
    plot_target_series(series)
    plot_rolling_statistics(series, 30)
    plot_acf_pacf(series, 40)
    data = create_returns(data, 'close')
    data = create_log_returns(data, 'close')
    data = create_lag_features(data, 'close', 5)
    data = create_rolling_features(data, 'close', (7, 30, 60))

    baseline_forecast = naive_forecasting(train, test)
    plot_forecast_model(test, baseline_forecast, 'naive')

    arima_model = fit_arima(train, (1, 1, 1))
    arima_predictions = forecast_arima(arima_model, len(test))

    sarimax_model = fit_sarimax(train, (1, 1, 1), (1, 0, 1, 5))
    sarimax_predictions = forecast_sarimax(sarimax_model, len(test))

    exp_smooth_model = fit_exponential_smoothing(train, 12, 'add', 'add')
    exp_smooth_predictions = forecast_exponential_smoothing(exp_smooth_model, len(test))

    plot_forecast_model(test, arima_predictions, 'arima')
    plot_forecast_model(test, sarimax_predictions, 'sarimax')
    plot_forecast_model(test, exp_smooth_predictions, 'exponential_smoothing')



    


if __name__ == '__main__':
    main()