import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / 'plots'
MODELS_DIR = PROJECT_ROOT / 'models'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
#  LOADING DATASET
# =========================

def load_data():
    cancer = load_breast_cancer()

    return cancer


# ===========================
#  EXPLANATORY DATA ANALYSIS
# ===========================

def eda(cancer):
    print(f'\n ===== KEYS OF THE DATASET ===== \n{cancer.keys()}')
    print(f'\n ===== DESCRIPTION OF THE DATASET =====\n {cancer.DESCR}')

    
    print(f'\n ===== SHAPE OF THE DATASET ===== \n{cancer.data.shape}')
    print(f'\n ===== TARGET NAMES ===== \n {cancer.target_names}')
    print(f'\n ===== FEATURE NAMES ===== \n{cancer.feature_names}')


# =========================
#  CREATING DATAFRAME
# =========================

def dataframe(cancer):
    cancer_df = cancer.copy()
    cancer_df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

    print(f'\n ===== WINCONSIN BREAST CANCER DATASET ===== \n{cancer_df.head(5)}')

    return cancer_df


# =========================
#  FEATURES SCALING
# =========================

def scale_features(cancer_df):
    scaler = StandardScaler()

    scaled_cancer = scaler.fit_transform(cancer_df)

    print(f'\n ===== SCALED DATASET ===== \n {scaled_cancer}')

    return scaler, scaled_cancer


# =========================
#  SELECTING OPTIMAL K 
# =========================

def optimal_k(scaled_cancer):
    inertias = []
    sil_score = []
    k_range = range(2, 10)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=10)
        labels = model.fit_predict(scaled_cancer)
        inertias.append(model.inertia_)
        scores = silhouette_score(scaled_cancer, labels)
        sil_score.append(scores)

        best_k = k_range[np.argmax(scores)]

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
#  TRAINING MODEL
# =========================

def train_model(scaled_cancer, best_k):

    model = KMeans(n_clusters=best_k, random_state=42, init='k-means++', n_init=10)
    labels = model.fit_predict(scaled_cancer)

    return model, labels



# ===============================
#  ADD CLUSTER LABELS TO DATASET
# ===============================

def add_cluster_labels(cancer_df, labels):

    cluster_cancer_df = cancer_df.copy()
    cluster_cancer_df['cluster'] = labels


    print(f'\n ===== DATASET WITH CLUSTER LABELS ===== \n{cluster_cancer_df}')

    return cluster_cancer_df


# =============================
#  PCA TRANSFORMATION
# =============================

def pca(scaled_cancer):

    pca_model = PCA(n_components=2, random_state=42)
    reduced_pca = pca_model.fit_transform(scaled_cancer)

    print( f'\n ===== REDUCED DATASET DIMENSION ===== \n {reduced_pca}')

    return pca_model, reduced_pca


# =================================
#  PCA VISUALIZATION WITH CENTROIDS
# =================================

def plot_pca(reduced_pca, labels, model, best_k, pca_model):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=reduced_pca[:, 0], 
        y=reduced_pca[:, 1], 
        hue=labels, 
        palette='tab10', 
        s=80
    )
    centroids_pca = pca_model.transform(model.cluster_centers_)

    plt.scatter(centroids_pca[:, 0],
                centroids_pca[:, 1 ],
                marker='x',
                s=200,
                linewidths=3,
                label='centroids')
    
    plt.title(f'PCA Projection (k = {best_k})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(PLOTS_DIR/ 'pca_plot.png')
    plt.close()


# =========================
#  SAVING MODEL AND SCALER
# =========================
    
def save_model(model, scaler):

    joblib.dump(model, MODELS_DIR/ 'model.joblib')
    joblib.dump(scaler, MODELS_DIR/ 'scaler.joblib')




# =========================
#  MAIN
# =========================

def main():
    cancer = load_data()
    eda(cancer)
    cancer_df = dataframe(cancer)
    scaler, scaled_cancer = scale_features(cancer_df)
    best_k, scores = optimal_k(scaled_cancer)
    model, labels = train_model(scaled_cancer, best_k)
    cluster_cancer_df = add_cluster_labels(cancer_df, labels)
    pca_model, reduced_pca = pca(scaled_cancer)
    plot_pca(reduced_pca, labels, model, best_k, pca_model)
    save_model(model, scaler)



if __name__ == '__main__':
    main()