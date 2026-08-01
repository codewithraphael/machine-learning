import datetime
import joblib
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
PLOTS_DIR = PROJECT_ROOT / 'plots'


# =========================
#  LOADING DATASET
# =========================














# =========================
#  MAIN
# =========================

def main():

    pass



if __name__ == '__main__':
    main()