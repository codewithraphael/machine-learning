import random

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================
#  KMEANS IMPLEMENTATION
# =========================

def initialize_centroids(data, k, method='random'):
    n_samples, n_features = data.shape
    if method == 'random':
        random_indices = random.sample(range(n_samples), k)
        centroids = data[random_indices].copy()