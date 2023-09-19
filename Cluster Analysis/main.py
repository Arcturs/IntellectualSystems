import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_dendrogram(data, labels):
    distance_matrix = linkage(data.to_numpy(), method='ward', metric='euclidean')
    print(f'Distance matrix: {distance_matrix}')

    fig = plt.figure(figsize=(15, 30))
    fig.patch.set_facecolor('white')

    dendrogram(
        distance_matrix,
        labels=labels,
        orientation='left',
        leaf_font_size=12)
    plt.title('Дендрограмма')
    return distance_matrix


def clusterization(data, cluster_info):
    fig, axes = plt.subplots(2, 2, figsize=(30, 45))
    axes[0, 0].scatter(
        data.to_numpy()[:, 0],
        data.to_numpy()[:, 3],
        c=cluster_info)
    axes[0, 0].set_title('', fontsize=16)
    axes[0, 0].set_xlabel('Средняя температура')
    axes[0, 0].set_ylabel('Средняя температура почвы')

    axes[0, 1].scatter(
        data.to_numpy()[:, 1],
        data.to_numpy()[:, 4],
        c=cluster_info)
    axes[0, 1].set_title('', fontsize=16)
    axes[0, 1].set_xlabel('Максимальная температура')
    axes[0, 1].set_ylabel('Месячные осадки')

    axes[1, 0].scatter(
        data.to_numpy()[:, 2],
        data.to_numpy()[:, 6],
        c=cluster_info)
    axes[1, 0].set_title('', fontsize=16)
    axes[1, 0].set_xlabel('Минимальная температура')
    axes[1, 0].set_ylabel('Неизвестная хар-ка')

    axes[1, 1].scatter(
        data.to_numpy()[:, 5],
        data.to_numpy()[:, 7],
        c=cluster_info)
    axes[1, 1].set_title('', fontsize=16)
    axes[1, 1].set_xlabel('Количество снежных дней')
    axes[1, 1].set_ylabel('Неизвестная хар-ка')


def create_kmeans_plots(data):
    wcss = []
    silh = []
    y_kmeans = []
    for i in range(2, 12):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=11, random_state=0)
        kmeans.fit(data)
        cluster_labels = kmeans.fit_predict(data)
        if i == 4:
            y_kmeans = cluster_labels
        wcss.append(kmeans.inertia_)
        silh.append(silhouette_score(data, cluster_labels, metric='euclidean'))

    plt.figure(figsize=(5, 5))
    plt.plot(range(2, 12), silh)
    plt.title('Индекс кластерного силуэта')
    plt.xlabel('Число кластеров')
    plt.ylabel('Silhouette score')

    plt.figure(figsize=(5, 5))
    plt.plot(range(2, 12), wcss)
    plt.title('Метод локтя')
    plt.xlabel('Число кластеров')
    plt.ylabel('WCSS')

    return y_kmeans


def __main__():
    file_path = pathlib.Path(__file__).with_name('data.csv')
    data = pd.read_csv(file_path, delimiter=";", usecols=[*range(1, 9)], header=None)
    labels = pd.read_csv(file_path, delimiter=";", usecols=[0], header=None).values
    print(f'Given data: {data}')
    print(data.describe())

    distance_matrix = create_dendrogram(data, labels)
    clusterization(data, fcluster(distance_matrix, 4, criterion='maxclust'))

    y_kmeans = create_kmeans_plots(data)
    clusterization(data, y_kmeans)

    plt.show()


if __name__ == '__main__':
    __main__()
