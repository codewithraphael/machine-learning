import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / 'plots'
DATA_PATH = PROJECT_ROOT / 'data'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)



# =========================
#  LOADING DATASET
# =========================

def load_data():
    iris = load_iris()

    X = iris.data
    y = iris.target

    iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

    print(f'\n ===== IRIS FLOWER DATASET ===== \n {iris_df.head(10)}')

    return iris, iris_df, X, y


# ===========================
#  EXPLANATORY DATA ANALYSIS
# ===========================

def eda(iris):
    print(f'\n ===== KEYS OF THE DATASET ===== \n{iris.keys()}')
    print(f'\n ===== DESCRIPTION OF THE DATASET =====\n {iris.DESCR}')

    
    print(f'\n ===== SHAPE OF THE DATASET ===== \n{iris.data.shape}')
    print(f'\n ===== TARGET NAMES ===== \n {iris.target_names}')
    print(f'\n ===== FEATURE NAMES ===== \n{iris.feature_names}')


# =========================
#  3D PCA PLOT
# =========================

def plot_3d(X, y):

    fig = plt.figure(1, figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=48, azim=134)

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k')

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean(),
                  X[y == label, 2].mean(), name)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        ax.set_title('3D PCA PLOT')

    plt.savefig(PLOTS_DIR / '3d_pca_plot.png')
    plt.close()


# =========================
#  FEATURE SCALING
# =========================

def scale_features(iris_df):
    scaler = StandardScaler()

    scaled_iris = scaler.fit_transform(iris_df)

    return scaled_iris, scaler


# =========================
#  OPTIMAL K
# =========================

def optimal_k(scaled_iris):

    inertias = []
    sil_score = []
    k_range = range(2, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, )
        labels = model.fit_predict(scaled_iris)
        inertias.append(model.inertia_)
        
        scores = silhouette_score(scaled_iris, labels)
        sil_score.append(scores)

        best_k = k_range[np.argmax(sil_score)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(k_range, inertias, marker='o', color='steelblue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertias')

    ax2.plot(k_range, sil_score, marker='o', color='coral')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'optimal_k.png')
    plt.close()

    print(f'\n ===== BEST K BASED ON SILHOUETTE SCORE ===== \n {best_k}')

    return best_k, scores


# =========================
#  FITTING MODEL
# =========================

def fit_model(scaled_iris):
    model = KMeans(n_clusters=3, random_state=42, init='k-means++', n_init=10)
    labels = model.fit_predict(scaled_iris)

    return model, labels


# ===========================
#  CLUSTER LABELS TO DATASET
# ===========================

def add_cluster_labels(iris_df, labels):

    clustered_data_path = PROJECT_ROOT / 'data' / 'clustered_iris_dataset.csv'
    cluster_iris_df = iris_df.copy()
    cluster_iris_df['cluster'] = labels

    print(f'\n ===== CLUSTER LABELS ON DATASET ===== \n {cluster_iris_df}')

    cluster_iris_df.to_csv(clustered_data_path)

    return cluster_iris_df



# =========================
#  PCA TRANSFORMATION
# =========================

def pca_transform(scaled_iris):

    pca = PCA(n_components=2, random_state=42)
    reduced_pca = pca.fit_transform(scaled_iris)

    return pca, reduced_pca


# =================================
#  PCA VISUALIZATION WITH CENTROIDS
# =================================

def plot_pca(pca, reduced_pca, model, labels, best_k):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=reduced_pca[:, 0],
        y=reduced_pca[:, 1],
        hue=labels,
        palette='tab10',
        s=80
    )
    centroid_pca = pca.transform(model.cluster_centers_)

    plt.scatter(
        centroid_pca[:, 0],
        centroid_pca[:, 1],
        marker='x',
        s=200,
        linewidths=3,
        label='centroids'
    )

    plt.title(f'PRINCIPAL COMPONENT PROJECTION (k = {best_k})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(PLOTS_DIR / 'pca_plot.png')
    plt.close()







# =========================
#  MAIN
# =========================

def main():
    iris, iris_df, X, y = load_data()
    eda(iris)
    plot_3d(X, y)
    scaled_iris, scaler = scale_features(iris_df)
    best_k, scores = optimal_k(scaled_iris)
    model, labels = fit_model(scaled_iris)
    cluster_iris_df = add_cluster_labels(iris_df, labels)
    pca, reduced_pca = pca_transform(scaled_iris)
    plot_pca(pca, reduced_pca, model, labels, best_k)



if __name__ == '__main__':
    main()