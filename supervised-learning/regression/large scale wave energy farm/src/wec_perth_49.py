import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / 'data/WEC/WEC_Perth_49.csv'
PLOTS_PATH = PROJECT_DIR / 'plots'
MODELS_PATH = PROJECT_DIR / 'models'

file_path = DATA_PATH


# ====================================
#  LOADING DATASET
# ====================================

def load_data(filepath):

    filepath = file_path

    if not filepath.exists():
        raise FileExistsError(f'File Not Found {filepath}')

    data = pd.read_csv(filepath)

    return data


# ====================================
#  DATA VALIDATION
# ====================================

def validate_data(data):

    print('='*100)
    print(' '*50 + 'WEC PERTH 49 ANALYSIS')
    print('='*100)

    print(data.head(5))

    print(f'\n ===== SHAPE OF DATASET ===== \n {data.shape}')
    print(f'\n ===== DATA TYPES ===== \n {data.dtypes}')
    print(f'\n ===== MISSING VALUES ===== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== DATA INFO ===== \n {data.info()}')
    print(f'\n ===== DUPLICATE VALUES ===== \n {data.duplicated().sum()}')
    print(f'\n ===== UNIQUE VALUES ===== \n {data.nunique()}')
    print(f'\n ===== SUMMARY STATISTICS ===== \n {data.describe()}')


# ====================================
#  DATA CLEANING
# ====================================

def clean_data(data):

    data = data.drop_duplicates()
    data = data.dropna()

    return data


# ====================================
# DATA VISUALIZATION 
# ====================================

def visualize_data(data):
    '''
    Data Visualization on Reactive Power and Total Power Distribution 
    '''

    plt.figure(figsize=(12, 6))
    sns.histplot(data['qW'], bins=30, kde=True)
    plt.title('Reactive Power Distribution')
    plt.xlabel('Reactive Power (kVAR)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_PATH / 'wec_perth_49_reactive_power_distribution.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(data['Total_Power'], bins=30, kde=True)
    plt.title('Total Power Distribution')
    plt.xlabel('Total Power (kW)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_PATH / 'wec_perth_49_total_power_distribution.png')
    plt.close()




# ====================================
#  MAIN
# ====================================

def main():

    data = load_data(file_path)
    validate_data(data)
    data = clean_data(data)
    visualize_data(data)


if __name__ == '__main__':
    main()