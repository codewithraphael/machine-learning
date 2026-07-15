import pandas as pd
import numpy as np

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
#  PAIRPLOT DISTRIBUTION
# =========================

def plot_pairplot(cancer_df):
    sns.pairplot(cancer_df)
    plt.savefig(PLOTS_DIR / 'pairplot_distribution.png')
    plt.close()

# =========================
#  FEATURES SCALING
# =========================

def scale_features(cancer_df):
    scaler = StandardScaler()

    scaled_cancer = scaler.fit_transform(cancer_df)

    print(f'\n ===== SCALED DATASET ===== \n {scaled_cancer}')

    return scaled_cancer


# =============================
#  PRINCIPAL COMPONENT ANALYSIS
# =============================

def pca(scaled_cancer):
    pca = PCA(n_components=2, random_state=42)
    reduced_pca = pca.fit_transform(scaled_cancer)

    print( f'\n ===== REDUCED DATASET DIMENSION ===== \n {reduced_pca}')

    return reduced_pca

# =========================
#  PLOTTING PCA 
# =========================

def plot_pca(reduced_pca, cancer):
    plt.figure(figsize=(8, 6))

    plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], c=cancer['target'])
    plt.xlabel('cluster 1')
    plt.ylabel('cluster 2')
    plt.savefig(PLOTS_DIR/ 'pca_plot.png')
    plt.close()



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
        sil_score.append(silhouette_score(scaled_cancer, labels))




# =========================
#  MAIN
# =========================

def main():
    cancer = load_data()
    eda(cancer)
    cancer_df = dataframe(cancer)
    plot_pairplot(cancer_df)
    scaled_cancer = scale_features(cancer_df)
    reduced_pca = pca(scaled_cancer)
    plot_pca(reduced_pca, cancer)


if __name__ == '__main__':
    main()