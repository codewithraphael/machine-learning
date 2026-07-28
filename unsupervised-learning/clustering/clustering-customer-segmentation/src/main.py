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
#  CHECKING FOR OUTLIERS
# =========================
def check_outliers(data):

    for col in ['Quantity', 'UnitPrice']:

        q1 = data[col].quantile(0.01)
        q3 = data[col].quantile(0.99)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        print(f'\n ===== Lower Bound ===== \n {lower_bound}')
        print(f'\n ===== Upper Bound ===== \n {upper_bound}')

        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]






'''
# =========================
#  RFM MODEL
# =========================
def rfm(data):

    recency_data = data.copy()

    
    Recency: The value of how recently a customer purchased at the establishment,
    Frequency: How frequent the customer’s transactions are at the establishment,
    Monetary value: The dollar (or pounds in our case) value of all the transactions that the customer made at the establishment
    

    # RECENCY
    
    For our use case, we will define the reference date as one day after the last transaction in our dataset.
    
    reference_date = recency_data.InvoiceDate.max()
    reference_date = reference_date + datetime.timedelta(days=1)
    print(reference_date)

    
    We will construct the recency variable as the number of days before the reference date when a customer
    last made a purchase. The following snippet of code will create this variable for us.
    
    recency_data['days_since_last_purchase'] = reference_date - recency_data.InvoiceDate
    recency_data['days_since_last_purchase_num'] = recency_data['days_since_last_purchase'].astype('timedelta64[ns]')


    customer_history_df = recency_data.groupby("CustomerID").min().reset_index()[['CustomerID', 'days_since_last_purchase_num']]
    customer_history_df.rename(columns={'days_since_last_purchase_num':'recency'},inplace=True)

    return recency_data, customer_history_df



def plot_recency(recency_data, customer_history_df):
    # Convert timedelta to numeric (days)
    x = customer_history_df.recency.dt.total_seconds() / (24 * 3600)
    mu = np.mean(x)
    sigma = math.sqrt(np.var(x))
    n, bins, patches = plt.hist(x, 1000, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('Recency in days')
    plt.ylabel('Number of transactions')
    plt.title(r'$\mathrm{Histogram\ of\ sales\ recency}\ $')
    plt.grid(True)
    plt.savefig(PLOTS_DIR / 'recency_distribution.png')
    plt.close()
    

'''





# =========================
#  MAIN
# =========================

def main():
    filepath = DATA_PATH
    data = load_data(filepath)
    eda(data)
    data = preprocess_data(data)
    check_outliers(data)















if __name__ == '__main__':
    main()