import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
#  MAIN
# ====================================

def main():
    pass







if __name__ == '__main__':
    main()