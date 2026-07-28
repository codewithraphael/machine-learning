import pandas as pd
import numpy as np
import datetime
import math
import joblib

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns; sns.set_theme()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
        'InvoiceNo': 'nunique', # Frequency
        'TotalPrice': 'sum' # Monetary value
    }).reset_index()

    print(rfm)


    # Renaming Columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Handling Monetary values of 0
    rfm['Monetary'].replace(0, 1)

    print(f'\n ===== RFM SUMMARY STATISTICS ===== \n {rfm.describe()}')


# =========================
#  HANDLING SKEWNESS
# =========================
def handle_skewness(rfm):

    '''
    Handling skewed distribution in RFM data using log transformation.
    '''
    print('='*60)
    print('HANDLING SKEWNESS')
    print('='*60)


    # Calculating skewness before transformation

    print(f'\n ===== SKEWNESS BEFORE TRANSFORMATION ===== \n')
    for col in ['Recency', 'Frequency', 'Monetary']:
        skew = rfm[col].skew()
        print(f'{col}: {skew:.3f}')


    # Applying Log transformation

    rfm_transformed = rfm.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        rfm_transformed[f'{col}_log'] = np.log1p(rfm[col])

    print(f'\n ===== SKEWNESS AFTER TRANSFORMATION ===== \n')
    for col in rfm_transformed:
        skew = rfm_transformed[col].skew()
        print(f' {col}: {skew:.3f}')

    return rfm_transformed








# =========================
#  MAIN
# =========================

def main():
    filepath = DATA_PATH
    data = load_data(filepath)
    eda(data)
    data = preprocess_data(data)
    handle_outliers(data)
    rfm(data)















if __name__ == '__main__':
    main()