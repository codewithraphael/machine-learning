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
#  MAIN
# =========================

def main():
    data, labels = generate_data()
    row_dist = calculate_distance(data, labels)



if __name__ == '__main__':
    main()