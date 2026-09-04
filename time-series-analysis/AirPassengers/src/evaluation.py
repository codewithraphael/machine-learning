import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from config import PLOTS_PATH

def  evaluate_forecast(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)

    print('='*70)
    print(' '*25 + 'EVALUATION METRICS')
    print('='*70)

    print(f"\n MAE:: {mae}")
    print(f"\n RMSE:: {rmse:.4f}")

    return {
        'MAE': mae,
        'RMSE': rmse
    }


def fit_arima(train, order):

    model = ARIMA(train,
                   order=order
    )

    arima_model = model.fit()

    print(arima_model.summary())

    return arima_model


def generate_forecast(model, steps):

    arima_predictions = model.forecast(
        steps = steps
    )

    return arima_predictions


def plot_forecast(actual, predicted):

    plt.figure(figsize=(16, 8))

    plt.plot(actual.index, actual, label='Actual')
    plt.plot(predicted.index, predicted, label='Predicted')

    plt.title('ARIMA Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_PATH / 'arima_forecast_vs_actual.png')
    plt.close()


def fit_sarima(train, order, seasonal_order):

    model = SARIMAX(
        train,
        order = order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarima_model = model.fit(disp=False)

    print(sarima_model.summary())

    return sarima_model


def model_diagnostics(model):

    model.plot_diagnostics(
        figsize=(20, 10)
    )

    plt.savefig(PLOTS_PATH / 'sarima_diagnostic_plot.png')
    plt.close()


def acf_plot(data, target_column):

    plot_acf(
        data[target_column].dropna(),
        lags=40
    )
    plt.savefig(PLOTS_PATH / 'acf_plot.png')
    plt.close()

def pacf_plot(data, target_column):

    plot_pacf(data[target_column],
              lags=40
    )
    plt.savefig(PLOTS_PATH / 'pacf_plot.png')
    plt.close()