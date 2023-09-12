import os.path
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def __main__():
    file_path = pathlib.Path(__file__).with_name('data.csv')
    data = pd.read_csv(file_path, delimiter=";", usecols=[*range(1, 8)], header=None)
    print(f'Given data: {data}')

    print(data.describe())
    distance_matrix = linkage(data.to_numpy(), method='ward', metric='euclidean')
    print(f'Distance matrix: {distance_matrix}')

    fig = plt.figure(figsize=(15, 30))
    fig.patch.set_facecolor('white')

    labels = pd.read_csv(file_path, delimiter=";", usecols=[0], header=None).values

    R = dendrogram(
        distance_matrix,
        labels=labels,
        orientation='left',
        leaf_font_size=12)

    # Возможно потом понадобится
    fig, axes = plt.subplots(1, 2, figsize=(15, 30))
    axes[0].scatter(
        data.to_numpy()[:,4],
        data.to_numpy()[:,1],
        c=fcluster(distance_matrix, 4, criterion='maxclust'))
    axes[0].set_title('', fontsize=16)
    plt.show()

    


if __name__ == '__main__':
    __main__()
