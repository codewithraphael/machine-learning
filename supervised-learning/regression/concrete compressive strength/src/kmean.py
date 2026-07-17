import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.spatial.distance import cdist, pdist

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


import warnings
warnings.filterwarnings('ignore')



# =========================
# LOADING DATASET 
# =========================

def load_data(filepath):
    
    data = pd.read_excel(filepath)

    return data


# ===========================
#  EXPLORATORY DATA ANALYSIS
# ===========================

def eda(data):
    shape = data.shape
    info = data.info()
    summary = data.describe()
    
    print(info)
    print(f'\n ===== Shape of Dataset ===== \n {shape}')
    print(f'\n ===== Summary Statistics ===== \n {summary}')


# =========================
#  FEATURE SCALING
# =========================

def scale_features(data):
    
    scaler = StandardScaler()
    x = scaler.fit_transform(data)

    return x


# =========================
#  OPTIMAL K VALUE
# =========================

def optimal_k_value(x):

    inertias = []
    sil_scores = []
    k_range = range(2, 10)

    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10,  random_state=42)
        labels = model.fit_predict(x)
        inertias.append(model.inertia_)
        sil_scores.append(silhouette_score(x, labels))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        ax1.plot(k_range, inertias, marker='o', color='green')
        ax1.set_title('Elbow Method')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertias')

        ax2.plot(k_range, sil_scores, marker='o', color='steelblue')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Scores')

        plt.tight_layout()
        plt.savefig('../plots/optimal_k.png')
        plt.close()

        best_k = k_range[sil_scores.index(max(sil_scores))]
        print(f'\n===== Best k value ===== \n\n {best_k}')

        return best_k











# =========================
#  MAIN FUNCTION
# =========================

def main():

    data = load_data('../data/concrete_data.xls')

    eda(data)

    x = scale_features(data)

    best_k = optimal_k_value(x)



if __name__ == '__main__':
    main()