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

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
#  LOADING DATASET
# =========================

def load_data():
    iris = load_iris()

    X = iris.data
    y = iris.target

    return iris, X, y


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
#  MAIN
# =========================

def main():
    iris, X, y = load_data()
    eda(iris)
    plot_3d(X, y)



if __name__ == '__main__':
    main()