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




# ====================
#  FEATURE ENGINEERING
# ====================

def feature_enginneering(data):

    # total pressure
    data['total_pressure'] = data['Lub oil pressure'] + data['Fuel pressure'] + data['Coolant pressure']

    # total temperature
    data['total_temp'] = data['lub oil temp'] + data['Coolant temp']

    # temperature difference
    data['temp_difference'] = data['lub oil temp'] - data['Coolant temp']
    data['pressure_difference'] = data['Lub oil pressure'] - data['Coolant pressure']
    
    # pressure difference
    data['temp_ratio'] = data['lub oil temp'] / data['Coolant temp']
    data['pressure_ratio'] = data['Lub oil pressure'] / data['Coolant temp']
    
    # rpm interaction
    data['rpm_oil_temp'] = data['Engine rpm'] * data['lub oil temp']
    data['rpm_coolant_temp'] = data['Coolant temp'] * data['Engine rpm']
    data['coolant_temp_per_rpm'] = data['Coolant temp'] / data['Engine rpm']
    data['oil_temp_per_rpm'] = data['lub oil temp'] / data['Engine rpm']

    # temperature categories
    data["oil_temp_category"] = pd.cut(
    data["lub oil temp"],
    bins=3,
    labels=["Low", "Normal", "High"])

    # pressure categories
    data["oil_pressure_status"] = pd.cut(
    data["Lub oil pressure"],
    bins=3,
    labels=["Low", "Normal", "High"])

    # Engine Load Score
    data['engine_load'] = ( 
        data['Engine rpm'] *
        data['Fuel pressure'] * 
        data['Coolant temp']
        ) / data['Lub oil pressure'].replace(0, np.nan)

    # load categories
    data["load_level"] = pd.cut(
    data["engine_load"],
    bins=3,
    labels=["Low", "Normal", "High"])


    return data



# =====================
#  DISPLAYING NEW DATA
# =====================
def display_new_data(data):
    print(f'\n ===== New Featured Data ===== \n {data}')


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

def plot_pairplot(data):
    sns.pairplot(data)
    plt.savefig('../plots/pairplot_distribution.png')
    plt.close()


# ====================
#  CORRELATION HEATMAP  
# ====================

def plot_correlation(data):
    corr_matrix = data.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, linewidths=0.5, cmap='viridis')
    plt.title('Features Correlation Heatmap')
    plt.savefig('../plots/features_correlation_plot.png')
    plt.close()


# ====================
#  MAIN FUNCTION
# ====================

def main():
    filepath = '../data/engine_data.csv'
    data = load_data(filepath)

    eda(data)
    data = feature_enginneering(data)
    display_new_data(data)

    plot_target(data)
    plot_pairplot(data)
    plot_correlation(data)


if __name__ == "__main__":
    main()