import os
import joblib
import warnings

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# =========================
#  LOADING DATASET
# =========================

def load_data(filepath):
    data = pd.read_csv(filepath)

    return data


# =========================
#  EXPLORATORY DATA ANALYSSI
# =========================

def eda(data):
    print("Data Shape:", data.shape)
    print("\nData Info:")
    print(data.info())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nDescriptive Statistics:")
    print(data.describe())


# =========================
#  PAIRPLOT DISTRIBUTION
# =========================

def plot_pairplot(data):
    
    plot = data.select_dtypes(include=[np.number])
    sns.pairplot(plot)
    plt.savefig('../plots/pairplot_distribution.png')
    plt.close()


# =========================
#  CORRELATION HEATMAP
# =========================
    
def plot_correlation_heatmap(data):

    corr_matrix = data.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, linewidth=0.5, cmap='jet')
    plt.savefig('../plots/heatmap_correlation.png')
    plt.close()


# =========================
#  NUMERIC FEATURE SELECTION
# =========================

def numeric_features(data):

    X = data.select_dtypes(include=[np.number])

    return X


# =========================
#  FEATURES SCALING
# =========================

def scale_features(X):

    scaler = StandardScaler()

    Scaled_X = scaler.fit_transform(X)

    return scaler, Scaled_X


# =========================
#  PLOT OPTIMAL K VALUE
# =========================

def plot_optimal_k(Scaled_X):

    inertias = []
    sil_scores = []
    k_range = range(1, 10)

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(Scaled_X)
        inertias.append(model.inertia_)
        sil_scores.append(silhouette_score(Scaled_X, labels))        
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(k_range, inertias, marker='o', color='steelblue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Numbers of clusters')
    ax1.set_ylabel('Inertias')

    ax2.plot(k_range, sil_scores, marker='o', color='coral')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Numbers of clusters')
    ax2.set_ylabel('Scores')

    plt.tight_layout()
    plt.savefig('../plots/optimal_k.png')
    plt.close()
















'''
# =========================
#  KMEANS IMPLEMENTATION
# =========================

def initialize_centroids(data, k, method='random'):
    n_samples, n_features = data.shape
    if method == 'random':
        random_indices = random.sample(range(n_samples), k)
        centroids = data[random_indices].copy()
'''


def main():

    data = load_data('../data/mtcars.csv')

    eda(data)

    plot_pairplot(data)
    plot_correlation_heatmap(data)

    X = numeric_features(data)
    scale_features(X)

    plot_optimal_k(X)



if __name__ == '__main__':
    main()