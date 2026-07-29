import pandas as pd
import numpy as np
import datetime
import joblib

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns; sns.set_theme()

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data/online retail.xlsx'
MODELS_DIR = PROJECT_ROOT / 'models'
PLOTS_DIR = PROJECT_ROOT / 'plots'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
#  LOADING CUSTOMERS DATASET
# ===========================

def load_data(filepath):

    data = pd.read_excel(filepath)

    return data


# =========================
#  EDA
# =========================

def eda(data):

    print(f'\n ===== Shape of the dataset ===== \n {data.shape}')
    print(f'\n =====Dataset Information ===== \n')
    print(data.info())
    print(f'\n ===== Checking Missing Values ===== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== Summary Statistics ===== \n {data.describe()}')

    id_country = data['Country'].value_counts().reset_index().head(5)
    print(f'\n ===== Index by country ===== \n {id_country}')

    unique_customers = data['CustomerID'].unique().shape
    percentage_of_orders = (data['CustomerID'].value_counts()/sum(data['CustomerID'].value_counts())*100) .head(n=13).cumsum()

    print(f'\n ===== Amount of Unique Customers ===== \n {unique_customers}')
    print(f'\n ===== Percentage Of Orders ===== \n {percentage_of_orders}')



# ===========================
#  DATA PREPROCESSING
# ===========================
def preprocess_data(data):

    '''
    Converting InvoiceDate to datetime,
    Removing rows with missing CustomerID,
    Removing Cancelled Orders (Invoice starting with 'c'),
    Removing negative quantites and prices,
    Feature Engineering: Calculating total price for each transaction,
    Removing outliers using IQR

    '''

    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


    initial_rows = len(data)
    data.dropna(subset=['CustomerID'], inplace=True)
    print(f'Removed {initial_rows - len(data)} rows with missing CustomerID')

    data = data[data['Quantity'] > 0]
    data = data[data['UnitPrice'] > 0]


    data['TotalPrice'] = data['Quantity']*data['UnitPrice']


    return data


# =========================
#  HANDLING OUTLIERS
# =========================
def handle_outliers(data):

    for col in ['Quantity', 'UnitPrice']:

        q1 = data[col].quantile(0.01)
        q3 = data[col].quantile(0.99)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        print(f'\n ===== Lower Bound ===== \n {col} : {lower_bound}')
        print(f'\n ===== Upper Bound ===== \n {col} : {upper_bound}')

        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    return data


# =========================
#  RFM MODEL
# =========================
def rfm(data):

    '''
    Recency: The value of how recently a customer purchased at the establishment,
    Frequency: How frequent the customer’s transactions are at the establishment,
    Monetary value: The dollar (or pounds in our case) value of all the transactions that the customer made at the establishment
    '''
    
    '''
    For our use case, we will define the reference date as one day after the last transaction in our dataset.
    '''

    reference_date = data.InvoiceDate.max()
    reference_date = reference_date + datetime.timedelta(days=1)
    print(f' Reference Date: {reference_date}')

    # Grouping by CustomerID to calculate RFM Metrics

    rfm_df = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique', # Frequency
        'TotalPrice': 'sum' # Monetary value
    }).reset_index()

    print(rfm_df)


    # Renaming Columns
    rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Handling Monetary values of 0
    rfm_df['Monetary'] = rfm_df['Monetary'].replace(0, 1)

    print(f'\n ===== RFM SUMMARY STATISTICS ===== \n {rfm_df.describe()}')


    return rfm_df


# =========================
#  HANDLING SKEWNESS
# =========================
def handle_skewness(rfm_df):

    '''
    Handling skewed distribution in RFM data using log transformation.
    '''
    print('='*60)
    print('HANDLING SKEWNESS')
    print('='*60)


    # Calculating skewness before transformation

    print(f'\n ===== SKEWNESS BEFORE TRANSFORMATION ===== \n')
    for col in rfm_df:
        skew = rfm_df[col].skew()
        print(f'{col}: {skew:.3f}')


    # Applying Log transformation

    rfm_transformed = rfm_df.copy()
    for col in rfm_transformed:
        rfm_transformed[f'{col}_log'] = np.log1p(rfm_df[col])

    print(f'\n ===== SKEWNESS AFTER TRANSFORMATION ===== \n')
    for col in rfm_transformed:
        skew = rfm_transformed[col].skew()
        print(f' {col}: {skew:.3f}')

    return rfm_transformed


# =============================
#  VISUALIZING RFM DISTRIBUTION
# =============================

def visualize_rfm_distributions(rfm_df, rfm_transformed):
    """
    Visualize RFM distributions before and after log transformation.
    
    Parameters:
    -----------
    rfm_df : pandas.DataFrame
        RFM dataframe with original and log-transformed columns
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    metrics = ['Recency', 'Frequency', 'Monetary']
    
    for i, metric in enumerate(metrics):
        # Original distribution
        axes[0, i].hist(rfm_df[metric], bins=50, edgecolor='black', alpha=0.7)
        axes[0, i].set_title(f'{metric} (Original)')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')

    for i, metric in enumerate(metrics):
        # Log-transformed distribution
        axes[1, i].hist(rfm_transformed[f'{metric}_log'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, i].set_title(f'{metric} (Log Transformed)')
        axes[1, i].set_xlabel('Log Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR /  'rfm_distributions.png', dpi=100, bbox_inches='tight')
    plt.close()



# =========================
#  FEATURE SCALING
# =========================
def scale_feature(rfm_df):

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    return x_scaled, scaler


# =========================
#  OPTIMAL K
# =========================
def optimal_k(x_scaled):

    '''
    Determinig optimal number of clusters
    '''

    inertia = []
    sil_scores = []
    k_range = range(2, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, init='k-means++', max_iter= 300, random_state=42)
        labels = model.fit_predict(x_scaled)

        inertia.append(model.inertia_)
        scores = silhouette_score(x_scaled, labels)
        sil_scores.append(scores)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(k_range, inertia, marker='o', color='steelblue')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')

    ax2.plot(k_range, sil_scores, marker='o', color='coral')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Scores')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'optimal_k.png')
    plt.close()





# =========================
# FITTING MODEL 
# =========================
def kmeans_clustering(x_scaled):

    '''
    Performing Kmeans clustering on scaled data
    '''

    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    k_labels = kmeans.fit_predict(x_scaled)

    return kmeans, k_labels


def hierarchical_clustering(x_scaled):

    '''
    Performing Hierarchical clustering
    '''

    hierarchical = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
    h_labels = hierarchical.fit_predict(x_scaled)

    return hierarchical, h_labels


# =========================
#  VISUALIZE CLUSTERS
# =========================
def visualize_pca(kmeans, x_scaled, k_labels):

    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(x_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = reduced_pca[:, 0],
                    y = reduced_pca[:, 1],
                    hue=k_labels,
                    palette='Set2',
                    legend='auto'
    )

    pca_centroids = pca.transform(kmeans.cluster_centers_)

    plt.scatter(pca_centroids[:, 0],
                pca_centroids[:, 1],
                c = 'red',
                marker = 'x',
                s=200
    )
    
    plt.title('PCA Visualization of Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.savefig(PLOTS_DIR / 'cluster_plot.png')
    plt.close()

def visualize_hierarchical(x_scaled):
    """
    Visualize hierarchical clustering using a dendrogram.
    """
    linkage_matrix = linkage(x_scaled, method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=8.
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'hierarchical_clustering.png', dpi=100, bbox_inches='tight')
    plt.close()


# ================================
#  PROFILE CLUSTERING AND PLOTTING
# ================================

def profile_clusters(rfm_df, labels):
    """
    Create a summary profile for each cluster using the RFM features.
    """
    cluster_df = rfm_df.copy()
    cluster_df['Cluster'] = labels

    cluster_summary = (
        cluster_df.groupby('Cluster')
        .agg(
            Customers=('CustomerID', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Min_Recency=('Recency', 'min'),
            Max_Recency=('Recency', 'max'),
            Min_Frequency=('Frequency', 'min'),
            Max_Frequency=('Frequency', 'max'),
            Min_Monetary=('Monetary', 'min'),
            Max_Monetary=('Monetary', 'max'),
        )
        .reset_index()
    )

    print('\n ===== CLUSTER PROFILES =====')
    print(cluster_summary)

    cluster_summary.to_csv(DATA_PATH / 'cluster_profiles.csv', index=False)
    return cluster_summary



# =========================
#  SAVING MODELS
# =========================


def save_model(model, filename):

    model_path = MODELS_DIR / filename
    joblib.dump(model, model_path)

# =========================
#  MAIN
# =========================
def main():
    filepath = DATA_PATH
    data = load_data(filepath)
    eda(data)
    data = preprocess_data(data)
    data = handle_outliers(data)
    rfm_df = rfm(data)
    rfm_transformed = handle_skewness(rfm_df)
    visualize_rfm_distributions(rfm_df, rfm_transformed)
    x_scaled, scaler = scale_feature(rfm_df)
    optimal_k(x_scaled)
    kmeans, k_labels = kmeans_clustering(x_scaled)
    hierarchical, h_labels = hierarchical_clustering(x_scaled)
    visualize_pca(kmeans, x_scaled, k_labels)
    visualize_hierarchical(x_scaled)
    profile_clusters(rfm_df, k_labels)
    save_model(kmeans, 'kmeans_model.joblib')
    save_model(hierarchical, 'hierarchical_model.joblib')


if __name__ == '__main__':
    main()