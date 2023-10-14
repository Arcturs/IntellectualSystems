import pandas as pd
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def find_predicted_values(k, x_train, x_test, y_train):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn.predict(x_test)


def __main__():
    file_path = pathlib.Path(__file__).with_name('vowel-context.data')
    data = pd.read_csv(file_path, header=None, delim_whitespace=True)
    y = data.iloc[:,1]
    x = data.drop(columns=[0, 1, 2, 13])
    x_train, x_test, y_train, y_test = train_test_split(x[528:], y[528:], test_size=0.25, random_state=42)

    best_k = None
    best_accuracy = 0
    for k in range(1, 13):
        y_pred = find_predicted_values(k, x_train, x_test, y_train)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print(f"Наилучшее значение k: {best_k}")

    y_pred = find_predicted_values(best_k, x_train, x_test, y_train)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Матрица сопряженности:")
    print(conf_matrix)

    error_rate = (1 - accuracy_score(y_test, y_pred)) * 100
    print(f"Процент ошибок: {error_rate:.2f}%")


if __name__ == '__main__':
    __main__()
