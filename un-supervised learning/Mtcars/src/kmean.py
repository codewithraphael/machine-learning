import warnings
from pathlib import Path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'mtcars.csv'
PLOTS_DIR = PROJECT_ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
#  LOADING DATASET
# =========================

def load_data(filepath):
    data = pd.read_csv(filepath)

    return data


# ============================
#  EXPLORATORY DATA ANALYSIS
# ============================

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
    plt.savefig(PLOTS_DIR / 'pairplot_distribution.png')
    plt.close()


# =========================
#  CORRELATION HEATMAP
# =========================
    
def plot_correlation_heatmap(data):

    corr_matrix = data.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, annot=True, linewidth=0.5, cmap='jet')
    plt.savefig(PLOTS_DIR / 'heatmap_correlation.png')
    plt.close()


# =========================
#  NUMERIC FEATURE SELECTION
# =========================

def numeric_features(data):

    X = data.select_dtypes(include=['int64', 'float64'])

    return X


# =========================
#  FEATURES SCALING
# =========================

def scale_features(X):

    scaler = StandardScaler()

    X_Scaled = scaler.fit_transform(X)

    print('\nMean after scaling: ', X_Scaled.mean(axis=0).round(2))
    print('\nStd after scaling: ', X_Scaled.std(axis=0).round(2))

    return scaler, X_Scaled


# ==================================
#  CREATING MULTIPLE CLUSTER LABELS
# ==================================

def multiple_cluster_labels(data, X_Scaled):

    for k in range(1, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        labels = model.fit_predict(X_Scaled)
        data[f'cluster_{k}'] = labels

        print(data)

    return data


# =========================
#  PLOT CLUSTERS
# =========================

def plot_clusters(data, X_Scaled):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(X_Scaled)

    fig, axes = plt.subplots(1, 5, figsize = (25, 5))

    for i, ax in enumerate(
        axes,
        start=2
    ):
        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c = data[f'cluster_{i}']
        )

        ax.set_title(
            f'cluster = {i}'
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'multiple_clusters.png')
    plt.close()
    



# =========================
#  PLOT OPTIMAL K VALUE
# =========================

def plot_optimal_k(X_Scaled):

    inertias = []
    sil_scores = []
    k_range = range(2, 10)

    best_k = None
    labels = None
    best_score = -1

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            init='k-means++',
            n_init=10
        )

        labels = model.fit_predict(X_Scaled)
        inertias.append(model.inertia_)
        score = silhouette_score(X_Scaled, labels)
        sil_scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k
            labels = labels

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
    plt.savefig(PLOTS_DIR / 'optimal_k.png')
    plt.close()

    print(f'\n===== Best K ===== \n {best_k}')

    return best_k, labels


# =========================
#  PROFILE CLUSTERING
# =========================

def profile_clusters(data, X_Scaled, best_k=5):

    profile_columns = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    profile = (
        data.groupby(f'cluster_{best_k}')[profile_columns].mean().round(1)
    )

    print(profile)

    return profile


# =========================
#  PLOT PROFILE HEATMAP
# =========================

def plot_profile_heatmap(profile):

    plt.figure(figsize=(12, 6))

    sns.heatmap(profile,
                annot=True,
                cmap='viridis',
                fmt='.1f',
                linewidths=0.5)
    
    plt.title('Cluster Profile')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'profile_cluster.png')
    plt.close()


# =========================
# SAVING CLUSTER OUTPUT
# =========================

def save_outputs(data, profile):
    clustered_data_path = PROJECT_ROOT / 'data' / 'mtcars_with_clusters.csv'
    profile_path = PROJECT_ROOT / 'data' / 'cluster_profile.csv'

    data.to_csv(clustered_data_path, index=False)
    profile.to_csv(profile_path)


# =========================
#  MAIN FUNCTION
# =========================

def main():

    data = load_data(DATA_PATH)
    data

    eda(data)

    plot_pairplot(data)
    plot_correlation_heatmap(data)

    X = numeric_features(data)
    scaler, X_Scaled = scale_features(X)

    data = multiple_cluster_labels(data, X_Scaled)

    plot_clusters(data, X_Scaled)


    best_k, labels = plot_optimal_k(X_Scaled)

    profile = profile_clusters(data, X_Scaled, best_k)
    plot_profile_heatmap(profile)
    save_outputs(data, profile)
    

if __name__ == '__main__':
    main()