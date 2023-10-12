import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


def paint_default_data(time_series):
    plt.figure(figsize=(12, 6))
    plt.plot(time_series)
    plt.title('Исходный временной ряд')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.show()


def paint_trend_and_seasonal(time_series):
    result = seasonal_decompose(time_series, model='additive', period=12)
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(result.trend)
    plt.title('Тренд')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(result.seasonal)
    plt.title('Сезонность')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def paint_forecast(time_series):
    p, d, q = 1, 1, 1  # Параметры ARIMA (порядок авторегрессии, интеграции, скользящего среднего)
    P, D, Q, s = 1, 1, 1, 12  # Параметры сезонности

    model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()

    # Выполнение прогноза на 12 месяцев вперед
    forecast_steps = 12
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_conf_int = forecast.conf_int()

    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Исходный ряд', color='blue')
    plt.plot(forecast.predicted_mean, label='Прогноз', color='red')
    plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink',
                     alpha=0.3)
    plt.title('Прогноз SARIMA')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    plt.show()

    forecast_values = forecast.predicted_mean[-forecast_steps:]
    print('Спрогнозированные значения на 12 месяцев вперед:')
    print(forecast_values)


def main():
    file_path = pathlib.Path(__file__).with_name('data.dat')
    data = pd.read_csv(file_path, delimiter="\t")

    time_series = data['spark']

    start_date = '01-01-1980'
    end_date = '07-01-1994'
    date_index = pd.date_range(start=start_date, end=end_date, freq='M')

    time_series.index = date_index
    paint_default_data(time_series)
    paint_trend_and_seasonal(time_series)
    paint_forecast(time_series)


if __name__ == '__main__':
    main()
