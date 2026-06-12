# %%

# ===== Loading Libraries =====

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_auc_score, roc_curve, RocCurveDisplay

from xgboost import XGBRegressor, XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# %%

# ===== Loading dataset =====

insurance = pd.read_csv('../dataset/Insurance claims data.csv')
insurance.head()

# %%

# ===== Summary Statistics =====
def summary_stat(data):
    shape = data.shape
    info = data.info()
    statistics = data.describe()

    print(f"\n ===== Shape of the dataset ===== \n {shape}")
    print(f"\n ===== Information about the dataset ===== \n {info}")
    print(f"\n ===== Summary Statistics ===== \n {statistics}")

summary_stat(insurance)

# %%

# ===== Handling Outliers
def handle_outliers(data):
    q1 = data.select_dtypes(include=['number']).quantile(0.25)
    q3 = data.select_dtypes(include=['number']).quantile(0.75)
    
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f'\n ===== Lower Bound ===== \n {lower_bound}')
    print(f'\n ===== Upper Bound ===== \n {upper_bound}')

    return lower_bound, upper_bound

lower_bound, upper_bound = handle_outliers(insurance)

# %%
# ===== Checking Outliers Counts, Bounds, Actual Min and Max Values
def check_outlier(data, lower_bound, upper_bound):
    for col in data.select_dtypes(include=['number']):
        mask = (lower_bound[col] < data[col]) | (upper_bound[col] > data[col]) 
        counts = mask.sum()
        print(f' {col}: {counts} outliers |'
              f'bounds = [{lower_bound.min():.3f} {upper_bound.max():.3f}] |'
              f'actual min = {lower_bound[col].min():.3f}, actual max = {upper_bound[col].max():.3f}'
              )

check_outlier(insurance, lower_bound=lower_bound, upper_bound=upper_bound)

# %%
# ===== Checking For Missing Values
def missing_values(data):
    values = data.isnull().sum().sort_values(ascending=False)

    print(f'\n ===== Missning Values ===== \n {values}') 

missing_values(insurance)

# %%
# ===== Visualizing the Outliers =====
figure = px.box(insurance.select_dtypes(include=['number']),
                orientation = 'h'
                )
figure.show()

# %%
