import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / 'plots'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ===================================
#  GENERATING SYNTHETIC BLOBS DATASET
# ===================================

def load_data():
    
    X, y = make_blobs(n_samples=500,
                      n_features=2,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=42)

    return X, y


# =========================
#  FEATURES SCALING
# =========================

def scale_features(X):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    return scaler, scaled_X



# =========================
#  OPTIMAL K
# =========================

def optimal_k(scaled_X):

    inertia = []
    sil_score = []
    k_range = range(2, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = model.fit_predict(scaled_X)
        inertia.append(model.inertia_)
        scores = silhouette_score(scaled_X, labels)
        sil_score.append(scores)

        best_k = k_range[np.argmax(sil_score)]

    fig, (ax1, ax2) = plt.subplots(1, 2 ,figsize=(13, 4))

    ax1.plot(k_range, inertia, marker='o', color = 'steelblue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')

    ax2.plot(k_range, sil_score, marker='o', color = 'coral')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Scores')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'optimal_k.png')
    plt.close()


    print(f'\n ===== DISTORTION ===== \n {model.inertia_:.2f}')
    print(f'\n ===== BEST K VALUE ===== \n {best_k}')

    return best_k


# =========================
#  FITTING MODEL
# =========================

def fit_model(scaled_X, best_k):
    model = KMeans(n_clusters=best_k, random_state=42, init='k-means++', n_init=10)
    labels = model.fit_predict(scaled_X)

    return model, labels



# =========================
#  DATA VISUALIZATION
# =========================
def plot_scatterplot(scaled_X, y, model, labels):
    sns.scatterplot(x = scaled_X[:, 0], 
                    y = scaled_X[:, 1], 
                    hue=y, 
                    legend='auto'
    )

    plt.scatter(model.cluster_centers_[:, 0], 
                model.cluster_centers_[:, 1], 
                c='red', 
                marker='x', 
                label='centroids'
    )

    plt.legend()
    plt.savefig(PLOTS_DIR / 'optimal_cluster.png')
    plt.close()





# =========================
#  MAIN
# =========================
def main():
    X, y = load_data()
    scaler, scaled_X = scale_features(X)
    best_k = optimal_k(scaled_X)
    model, labels = fit_model(scaled_X, best_k)
    plot_scatterplot(scaled_X, y, model, labels)



if __name__ == '__main__':
    main()

