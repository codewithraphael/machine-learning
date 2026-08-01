import joblib
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
PLOTS_DIR = PROJECT_ROOT / 'plots'


# =========================
#  LOADING DATASET
# =========================

def load_data(filepath):

    hourly_data = pd.read_csv(filepath)

    return hourly_data



# ============================
#  EXPLANATORY DATA ANALYSIS
# ============================
def eda(hourly_data):
    print('='*60)
    print('EXPLANATORY DATA ANALYSIS')
    print('='*60)

    print(f'\n ===== SHAPE OF DATASET ===== \n {hourly_data.shape}')
    print('===== DATASET INFORMATION ===== \n')
    print(hourly_data.info())
    print(f'\n ===== SUMMARY STATISTICS ===== \n {hourly_data.describe()}')
    print(f'\n ===== MISSING VALUE ===== \n {hourly_data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== DUPLICATE DATA ===== \n {hourly_data.duplicated().sort_values(ascending=False)}')


# =========================
#  DATA PREPROCESSING
# =========================
def preprocess_data(hourly_data):

    '''
    renaming columns,
    converting date to datetime,
    converting suitable datatypes to categorical data
    '''

    hourly_data.rename(columns={
        'instant': 'rec_id',
        'dteday': 'date',
        'holiday': 'is_holiday',
        'workingday': ' is_workingday',
        'weathersit': 'weather_condition',
        'hum': 'humidity',
        'mnth': 'month',
        'cnt': 'total_count',
        'hr': 'hour',
        'yr': 'year'
    }, inplace=True)


    hourly_data['date'] = pd.to_datetime(hourly_data['date'])

    cat_columns = ['season',
                   'is_holiday',
                   'weekday',
                   'weather_condition',
                   'is_workingday',
                   'month',
                   'year',
                   'hour'
    ]

    for column in cat_columns:
        hourly_data[column] = hourly_data[column].astype('category')

        print(f'\n ===== CHECKING DTYPES ===== \n {hourly_data.dtypes}')
    
    return hourly_data







# =========================
#  MAIN
# =========================

def main():

    filepath = DATA_PATH / 'hour.csv'
    hourly_data = load_data(filepath)
    eda(hourly_data)
    preprocess_data(hourly_data)



if __name__ == '__main__':
    main()