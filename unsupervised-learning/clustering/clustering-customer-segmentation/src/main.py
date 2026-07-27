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


# =========================
#  FEATURE ENGINEERING
# =========================
def feat_eng(data):
    
    data['Amount'] = data['Quantity']*data['UnitPrice']
    print(data)

    return data

# ===========================
#  DATA CLEANING & SELECTION
# ===========================
def select_data(data):

    '''
    Selecting UK Customers, which are notably the largest segment(based on country)
    &
    Removing negative amount or return transactions
    '''
    data = data[data.Country == 'United Kingdom']

    data = data[~(data.Amount<0)]

    print(data.head(5))
    return data


# =========================
#  REMOVING NULL VALUES
# =========================
def remove_null(data):

    data = data[~(data.CustomerID.isnull())]
    print(data.shape)

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

    # RECENCY
    reference_date = data.InvoiceDate.max()
    reference_date = reference_date + datetime.timedelta(days=1)
    print()

    pass




# =========================
#  MAIN
# =========================

def main():
    filepath = DATA_PATH
    data = load_data(filepath)
    eda(data)
    data = feat_eng(data)
    data = select_data(data)
    data = remove_null(data)














if __name__ == '__main__':
    main()