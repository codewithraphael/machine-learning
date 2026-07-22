import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage

import warnings
warnings.filterwarnings('ignore')


# ============================
#  GENERATING RANDOM DATASET
# ============================

def generate_data():
    np.random.seed(123)
    variables = ['x', 'y', 'z']
    labels = ['id_0', 'id_1', 'id_2', 'id_3', 'id_4']
    X = np.random.random_sample([5, 3])*10

    data = pd.DataFrame(X, columns=variables, index=labels)
    print(f'\n ===== DATASET ===== \n {data}')

    return data, labels


# =============================
#  CALCULATING DISTANCE MATRIX
# =============================

def calculate_distance(data, labels):
    
    row_dist = pd.DataFrame(
        squareform(
            pdist(data, metric='euclidean')
        ), columns=labels, index=labels
    )

    print(f'\n ===== ROW DISTANCE ===== \n {row_dist}')

    return row_dist



# =========================
#  COMPUTING ROW CLUSTERS
# =========================

def compute_row_clusters(data, row_dist):

    row_clusters = linkage(pdist(
        data,
        method='complete',
        metric='euclidean')
    )

    pd.DataFrame(row_clusters, 
                 columns=['row label 1',
                          'row label 2',
                          'distance',
                          'no. of items in clust.'], 
                          index=[f'cluster {(i + 1)}' for i in range (row_clusters.shape[0])])








# =========================
#  MAIN
# =========================

def main():
    data, labels = generate_data()
    row_dist = calculate_distance(data, labels)
    compute_row_clusters(data, row_dist)


if __name__ == '__main__':
    main()