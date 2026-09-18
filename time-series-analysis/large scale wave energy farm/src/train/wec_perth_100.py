import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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


# ===========================================
#  FEATURE SELECTION & PREPROCESSING PIPELINE
# ===========================================
def preprocess_data(data):

    data = data.rename(columns={
        'Total_Power': 'total_power',
        'qW' : 'farm_q_factor'
    })

    scaler = StandardScaler()
    features = data.drop(columns=['total_power'], axis=1)
    target = data['total_power']

    X = features
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test


# ====================================
#  MODEL TRAINING PIPELINE
# ====================================
def train_model(X_train, y_train):

    models = {
        'LINEAR REGRESSION' : LinearRegression(),
        'RANDOMFOREST REGRESSOR' : RandomForestRegressor(),
        'GRADIENT BOOSTING REGRESSOR' : GradientBoostingRegressor(),
        'XGBOOST REGRESSOR' : XGBRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# ====================================
#  MODEL EVALUATION
# ====================================
def evaluate_model(X, y, X_train, X_test, y_train, y_test, name, model):

    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)    
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X, y, cv=skf, scoring='r2')

    print('='*100)
    print(f'{name}')
    print('='*100)

    print(f'\n ===== TRAINING SCORE ===== \n {train_score:.3f}')
    print(f'\n ===== TEST SCORE ===== \n {test_score:.3f}')
    print(f'\n ===== R2 SCORE ===== \n {r2:.3f}')
    print(f'\n ===== MEAN ABSOLUTE ERROR ===== \n {mae:.3f}')
    print(f'\n ===== MEAN SQUARED ERROR ===== \n {mse:.3f}')
    print(f'\n ===== ROOT MEAN SQUARED ERROR ===== \n {rmse:.3f}')
    print(f'\n ===== CROSS VALIDATION SCORES ===== \n {cv}')
    print(f'\n ===== CROSS VALIDATION MEAN & STD. ===== \n {cv.mean():.3f} (+/-) {cv.std()*2:.3f}')




# ====================================
#  MAIN
# ====================================

def main():

    filepath = DATA_PATH / 'WEC_Perth_100.csv'
    data = load_data(filepath)
    eda(data)
    visualize_data(data)
    X, y, X_train, X_test, y_train, y_test = preprocess_data(data)
    trained_models = train_model(X_train, y_train)

    for (name, model) in (trained_models).items():
        evaluate_model(X, y, X_train, X_test, y_train, y_test, name, model)





if __name__ == '__main__':
    main()