import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


def main():
    file_path = pathlib.Path(__file__).with_name('data.dat')
    data = pd.read_csv(file_path, delimiter="\t", header=None, skiprows=1,
                       names=['fort', 'dry', 'sweet', 'red', 'rose', 'spark', 'total', 'year_', 'month_', 'date_'])

    # Временной ряд потребления всех вин
    time_series = data['total']

    # Даты начала и конца временного ряда
    # Индекс содержит даты между start_date и end_date с шагом в месяц
    start_date = '01-01-1980'
    end_date = '07-01-1994'
    date_index = pd.date_range(start=start_date, end=end_date, freq='M')

    time_series.index = date_index

    # Построение графика исходного временного ряда
    plt.figure(figsize=(12, 6))
    plt.plot(time_series)
    plt.title('Исходный временной ряд')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.show()

    # Использование метода seasonal_decompose для анализа сезонности(разложения временного ряда на компоненты тренда и сезонности).
    result = seasonal_decompose(time_series, model='additive', period=12)  # Период сезонности 12 месяцев
    #график тренда
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(result.trend)
    plt.title('Тренд')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)
    # график сезонности
    plt.subplot(212)
    plt.plot(result.seasonal)
    plt.title('Сезонность')
    plt.xlabel('Год')
    plt.ylabel('Значение')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    p, d, q = 1, 1, 1  # Параметры ARIMA (порядок авторегрессии, интеграции, скользящего среднего)
    P, D, Q, s = 1, 1, 1, 12  # Параметры сезонности

    # Объект SARIMA обучается на данных временного ряда.
    model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()

    # Выполнение прогноза на 8 месяцев вперед
    forecast_steps = 8
    forecast = results.get_forecast(steps=forecast_steps)

    forecast_conf_int = forecast.conf_int()

    # Построение графиков исходного ряда и прогноза
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

    # Вывод спрогнозированных значений
    forecast_values = forecast.predicted_mean[-forecast_steps:]
    print('Спрогнозированные значения на 8 месяцев вперед:')
    print(forecast_values)


if __name__ == '__main__':
    main()
