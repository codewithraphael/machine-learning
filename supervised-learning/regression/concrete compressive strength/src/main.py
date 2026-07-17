import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, silhouette_score
from sklearn.decomposition import PCA

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

import joblib

PLOTS_DIR = Path(__file__).resolve().parent.parent / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# =========================
#  LOADING DATASET
# =========================

def load_data(filepath):
    data = pd.read_excel(filepath)

    return data


# ===========================
#  EXPLORATORY DATA ANALYSIS
# ===========================

def eda(data):
    shape = data.shape
    info = data.info()
    summary = data.describe()
    
    print(info)
    print(f'\n ===== Shape of Dataset ===== \n {shape}')
    print(f'\n ===== Summary Statistics ===== \n {summary}')


# =========================
#  PAIRPLOT DISTRIBUTION
# =========================

def plot_pairplot(data):
    sns.pairplot(data)
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'plots', 'pairplot_distribution.png'))
    plt.close()


# =========================
# CORRELATION HEATMAP 
# =========================

def plot_correlation_heatmap(data):
    plt.figure(figsize=(18, 18))
    sns.heatmap(data.corr(), annot=True, linewidths=0.5, cmap='viridis', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'plots', 'correlation_heatmap.png'))
    plt.close()


def save_regression_plots(name, y_test, y_pred):
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    safe_name = name.lower().replace(' ', '_')

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name} - Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{safe_name}_actual_vs_predicted.png'))
    plt.close()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{name} - Residuals vs Fitted')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{safe_name}_residuals_vs_fitted.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{name} - Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{safe_name}_residual_distribution.png'))
    plt.close()


# ============================
# REGRESSION DIAGNOSTIC PLOTS
# ============================

def plot_regression_diagnostics(name, y_test, y_pred):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    residuals = y_test - y_pred

    filename_prefix = name.lower().replace(' ', '_')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.7)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{name} - Actual vs Predicted')

    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'{name} - Residuals')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'{filename_prefix}_diagnostics.png')
    plt.close(fig)

    hist_fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title(f'{name} - Residual Distribution')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    hist_fig.tight_layout()
    hist_fig.savefig(PLOTS_DIR / f'{filename_prefix}_residual_histogram.png')
    plt.close(hist_fig)


# ==========================================
# FEATURE SELECTION & PREPROCESSING PIPELINE
# ==========================================

def preprocess(data):

    scaler = StandardScaler()
    features_column = data.drop(columns=['concrete_compressive_strength'], axis=1)
    target_column = data['concrete_compressive_strength']
    
    X = features_column
    y = target_column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test


# =========================
# MODEL TRAINING
# =========================

def train(X_train, y_train):
    
    models = {
        'LINEAR REGRESSION': LinearRegression(),
        'RANDOMFOREST REGRESSOR': RandomForestRegressor(),
        'GRADIENT BOOSTING REGRESSOR': GradientBoostingRegressor(),
        'XGBOOST REGRESSOR': XGBRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# =========================
#  MODEL EVALUATION
# =========================

def evaluate(X, y, X_train, X_test, y_train, y_test, trained_models):
   for (name, model) in (trained_models.items()):
        y_pred = model.predict(X_test)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
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
        print(f'\n ===== MEAN ABSOLUTE ERROR ===== \n {mae:.3f}')
        print(f'\n ===== MEAN SQUARED ERROR ===== \n {mse:.3f}')
        print(f'\n ===== ROOT MEAN SQUARED ERROR ===== \n {rmse:.3f}')
        print(f'\n ===== CROSS VALIDATION SCORES ===== \n {cv}')
        print(f'\n ===== CROSS VALIDATION MEAN & STD. ===== \n {cv.mean():.3f} (+/-) {cv.std()*2:.3f}')


# ============================================
#  SAVING XGBOOST MODEL FOR DEPLOYMENT TESTING
# ============================================gg

def save_model(trained_models):
    for name, model in trained_models.items():
        if name == 'XGBOOST REGRESSOR':
            joblib.dump(model, '../models/xgboost_model.joblib')


# =========================
#  MAIN FUNCTION
# =========================

def main():

    filepath = '../data/concrete_data.xls'
    data = load_data(filepath)

    eda(data)

    plot_pairplot(data)
    plot_correlation_heatmap(data)

    X, y, X_train, X_test, y_train, y_test = preprocess(data)
    trained_models = train(X_train, y_train)

    print('\n===== TRAINED MODELS =====')
    print(list(trained_models.keys()))

    evaluate(X, y, X_train, X_test, y_train, y_test, trained_models)
    save_model(trained_models)

if __name__ == "__main__":
    main()