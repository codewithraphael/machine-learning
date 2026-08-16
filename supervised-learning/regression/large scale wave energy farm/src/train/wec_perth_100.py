import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data/WEC'
MODEL_PATH = PROJECT_ROOT / 'models'
PLOTS_PATH = PROJECT_ROOT / 'plots'

DATA_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)


# ====================================
#  LOADING DATASET
# ====================================
def load_data(filepath):

    data = pd.read_csv(filepath)

    return data


# ====================================
#  EXPLANATORY DATA ANALYSIS
# ====================================
def eda(data):

    print('='*80)
    print('WAVE ENERGY CONVERTERS PERTH 100')
    print('='*80)

    print(f' \n SHAPE OF THE DATASET :: {data.shape}')

    print('='*60)
    print('DISPLAYING DATASET')
    print('='*60)
    print(data.head(n=10))

    print('='*60)
    print('SUMMARY STATISTICS')
    print('='*60)
    print(data.describe())

    print('='*60)
    print('CHECKING FOR MISSING VALUES')
    print('='*60)
    print(data.isnull().sum().sort_values(ascending=False))


# ====================================
#  DATA VISUALIZATION
# ====================================
def visualize_data(data):

    '''
    Data Visualization on Reactive Power and Total Power Distribution 
    '''
    plt.figure(figsize=(12, 6))
    sns.histplot(data,
                 x=data['Total_Power'],
                 kde=True
    )
    plt.title('Total Power Distribution')
    plt.xlabel('Total Power (kW)')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'total_power_distribution.png')
    plt.close()


    sns.histplot(data,
                x=data['qW'],
                kde=True
    )
    plt.title('Reactive Power Distribution')
    plt.xlabel('Reactive Power (kVAR)')
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'reactive_power_distribution.png')
    plt.close()


# ====================================
#  MAIN
# ====================================

def main():

    filepath = DATA_PATH / 'WEC_Perth_100.csv'
    data = load_data(filepath)
    eda(data)
    visualize_data(data)







if __name__ == '__main__':
    main()