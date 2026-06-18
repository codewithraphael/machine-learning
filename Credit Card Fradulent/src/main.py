# =============================
# IMPORTING LIBRARIES
# =============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix

from xgboost import XGBClassifier

import joblib

import warnings
warnings.filterwarnings('ignore')


# =============================
# LOADING DATASET
# =============================

def load_data():
    data = pd.read_csv('../data/credit_card_fraud.csv')

    return data


# =============================
# EXPLANATORY DATA ANALYSIS
# =============================

def eda(data):
    shape = data.shape
    info = data.info()
    missing_value = data.isnull().sum().sort_values(ascending= False)
    statistics = data.describe()

    print(f'\n ===== Shape of the dataset ===== \n {shape}')
    print(info)
    print(f'\n ===== Missing Values ===== \n {missing_value}')
    print(f'\n ===== Summary Statistics ===== \n {statistics}')


# =============================
# COUNTPLOTS
# =============================

def count_plot(data):
    sns.countplot(data, x = 'is_fraud')
    plt.title('Target Distribution')
    plt.savefig('../plots/target_distribution.png')
    plt.close()

# =============================
# CORRELATION HEATMAP
# =============================

def correlation_plot(data):
    corr_plot = data.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_plot, annot=True, linewidths=0.5, cmap='viridis')
    plt.title('Features Correlation Heatmap')
    plt.savefig('../plots/correlation_heatmap.png')
    plt.close()





# =============================
# MAIN FUNCTION
# =============================

def main():
    data = load_data()
    eda(data)
    count_plot(data)
    correlation_plot(data)

if __name__ == "__main__":
    main()
