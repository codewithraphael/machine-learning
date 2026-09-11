import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import PLOTS_PATH


def naive_forecasting(train, test):

    '''
    forecast future values by repeating last observed value.
    '''

    last_value = train.iloc[-1]

    forecast = pd.Series(
        last_value,
        index=test.index
    )

    return forecast


def fit_arima(train, order=(1, 1, 1)):

    model = ARIMA(
        train,
        order=order
    )

    arima_model = model.fit()

    print('='*100)
    print(' '*30 + 'ARIMA MODEL SUMMARY')
    print('='*100)

    print(arima_model.summary())

    return arima_model


def forecast_arima(model, steps):

    arima_predictions = model.forecast(
        steps=steps
    )

    return arima_predictions


def fit_sarimax(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 5)):

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarimax_model = model.fit()

    print('='*100)
    print(' '*30 + 'SARIMAX MODEL SUMMARY')
    print('='*100)

    print(sarimax_model.summary())

    return sarimax_model


def forecast_sarimax(model, steps):

    sarimax_predictions = model.forecast(
        steps=steps
    )

    return sarimax_predictions


def fit_exponential_smoothing(train, seasonal_periods=12, trend='add', seasonal='add'):

    model = ExponentialSmoothing(
        train,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal
    )

    exp_smooth_model = model.fit()

    print('='*100)
    print(' '*30 + 'EXPONENTIAL SMOOTHING MODEL SUMMARY')
    print('='*100)

    print(exp_smooth_model.summary())

    return exp_smooth_model


def forecast_exponential_smoothing(model, steps):

    exp_smooth_predictions = model.forecast(
        steps=steps
    )

    return exp_smooth_predictions


def plot_forecast_model(actual, predicted, model_name='forecast'):

    actual = pd.Series(actual).copy()
    predicted = pd.Series(predicted).copy()

    if actual.empty or predicted.empty:
        raise ValueError('actual and predicted data must not be empty')

    if isinstance(actual.index, pd.MultiIndex):
        actual.index = actual.index.get_level_values(-1)

    if isinstance(predicted.index, pd.MultiIndex):
        predicted.index = predicted.index.get_level_values(-1)

    if len(actual) != len(predicted):
        if len(predicted) == len(actual.index):
            predicted.index = actual.index
        else:
            raise ValueError(
                f'actual and predicted lengths do not match: {len(actual)} vs {len(predicted)}'
            )

    plt.figure(figsize=(16, 8))

    plt.plot(actual.index, actual, label='Actual')
    plt.plot(predicted.index, predicted, label='Predicted')

    safe_model_name = str(model_name).strip().lower().replace(' ', '_')
    plt.title(f'{model_name.title()} Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_PATH / f'{safe_model_name}_forecast_vs_actual.png')
    plt.close()


def plot_forecast(actual, predicted, model_name='forecast'):
    plot_forecast_model(actual, predicted, model_name=model_name)


