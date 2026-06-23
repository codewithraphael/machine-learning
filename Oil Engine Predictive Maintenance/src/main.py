# ====================
# IMPORTING LIBRARIES 
# ====================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

import joblib

import warnings
warnings.filterwarnings('ignore')

# ====================
#  LOADING DATASET
# ====================

def load_data(filepath):
    data = pd.read_csv(filepath)

    return data

# ==========================
#  EXPLANATORY DATA ANALYSIS
# ==========================

def eda(data):
    shape = data.shape
    info = data.info()
    missing_value = data.isnull().sum().sort_values(ascending=False)
    statistics = data.describe()

    print(f'\n ===== Shape of the data ===== \n {shape}')
    print(info)
    print(f'\n ===== Missing Values ===== \n {missing_value}')
    print(f'\n ===== Summary Statistics ===== \n {statistics}')


# ==========================
#  TARGET DISTRIBUTION PLOT
# ==========================

def plot_target(data):
    plt.figure(figsize=(15, 5))
    sns.histplot(data, x = data['Engine Condition'])
    plt.savefig('../plots/engine_condition_distribution.png')
    plt.close()


# =======================
#  PAIRPLOR DISTRIBUTION
# =======================

def pairplot(data):
    sns.pairplot(data)
    plt.savefig('../plots/pairplot_distribution.png')
    plt.close()    

# ====================
#  MAIN FUNCTION
# ====================

def main():
    filepath = '../data/engine_data.csv'
    data = load_data(filepath)

    eda(data)
    plot_target(data)
    pairplot(data)


if __name__ == "__main__":
    main()