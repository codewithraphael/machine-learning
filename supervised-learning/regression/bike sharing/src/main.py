import joblib
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

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
    print(f'\n ===== DUPLICATE DATA ===== \n {hourly_data.duplicated().sum()}')


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
        'workingday': 'is_workingday',
        'weathersit': 'weather_condition',
        'hum': 'humidity',
        'mnth': 'month',
        'cnt': 'total_count',
        'hr': 'hour',
        'yr': 'year'
    }, inplace=True)


    hourly_data['date'] = pd.to_datetime(hourly_data['date'], format='%Y-%m-%d')

    cat_columns = ['season',
                   'is_holiday',
                   'weekday',
                   'weather_condition',
                   'is_workingday',
                   'month',
                   'year',
    ]

    for column in cat_columns:
        hourly_data[column] = hourly_data[column].astype('category')

    print(f'\n ===== DATA TYPES ===== \n {hourly_data.dtypes}')
    
    return hourly_data



# =========================
#  DATA VISUALIZATION
# =========================
def visualize_data(hourly_data):

    '''
    distribution & trends visualization for season, weekdays and monthly bike sharing data
    '''

    # hourly distribution of counts
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.pointplot(data=hourly_data[['hour', 'total_count', 'season']],
                  x='hour',
                  y='total_count',
                  hue='season',
                  ax=ax
    )
    ax.set_title('season wise hourly distribution of counts')
    plt.savefig(PLOTS_DIR / 'hourly_distribution_of_counts.png')
    plt.close()


    # monthly distribution of counts
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.barplot(data=hourly_data[['month', 'total_count']],
                    x='month',
                    y='total_count',
                    ax=ax
    )
    ax.set_title('monthly distribution of counts')
    plt.savefig(PLOTS_DIR / 'monthly_distribution_of_counts.png')
    plt.close()


    # yearly distribution of counts
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.violinplot(data=hourly_data[['year', 'total_count']],
                    x='year',
                    y='total_count',
                    ax=ax
    )
    ax.set_title('yearly distribution of counts')
    plt.savefig(PLOTS_DIR / 'yearly_distribution_of_counts.png')
    plt.close()


    # target column distribution analysis
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.histplot(data=hourly_data,
                 x='total_count',
                 kde=True,
                 ax=ax
    )
    ax.set_title('distribution of total bike rentals')
    ax.set_xlabel('total bike rentals')
    ax.set_ylabel('frequency')
    plt.savefig(PLOTS_DIR / 'target_distribution')
    plt.close()

    # exterme values / outliers visualization on target column
    fig, ax = plt.subplots(figsize=(22, 10))
    sns.boxplot(data=hourly_data,
                x='total_count',
                ax=ax
    )
    ax.set_title('boxplot of total bike rentals')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'target_boxplot.png')
    plt.close()



# ==============================
#  CORRELATION HEATMAP ANALYSIS
# ==============================
def plot_heatmap(hourly_data):

    '''
    correlation heatmap visualization for numerical features
    '''
    corr_matrix = hourly_data.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(22, 10))
    sns.heatmap(corr_matrix, annot=True, linewidths=0.5, cmap='viridis', ax=ax)
    ax.set_title('correlation heatmap')
    plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()


# =========================
#   FEATURE SELECTION
# =========================
def select_features(hourly_data):

    '''
    selecting features for regression model
    '''

    X = hourly_data.drop(columns=['total_count', 'casual', 'registered', 'date'])
    y = hourly_data['total_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


# ===================================
#  FEATURES ENCODING PIPELINE
# ===================================
def model_pipeline(X):

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    return preprocessor


# =========================
#  MODEL TRAINING PIPELINE
# =========================
def train_model(preprocessor, X_train, y_train):

    models = {
        'LINEAR REGRESSION': LinearRegression(),
        'RANDOMFOREST REGRESSOR': RandomForestRegressor(),
        'XGBOOST REGRESSOR': XGBRegressor()
    }

    trained_models = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        pipe.fit(X_train, y_train)

        trained_models[name] = pipe

    return trained_models



# =========================
#  MODEL EVALUATION
# =========================
def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):

    y_pred = pipe.predict(X_test)
  
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f'='*60)
    print(f'\n{name}')
    print(f'='*60)
    print(f'\n ===== TRAINING SCORE ===== \n {train_score:.3f}')
    print(f'\n ===== TEST SCORE ===== \n {test_score:.3f}')
    print(f'\n ===== R2 SCORE ===== \n {r2:3f}')
    print(f'\n ===== MEAN ABSOLUTE ERROR ===== \n {mae:.3f}')
    print(f'\n ===== MEAN SQUARED ERROR ===== \n {mse:.3f}')
    print(f'\n ===== ROOT MEAN SQUARED ERROR ===== \n {rmse:.3f}')

    return y_pred



# =========================
#  MAIN
# =========================

def main():

    filepath = DATA_PATH / 'hour.csv'
    hourly_data = load_data(filepath)
    eda(hourly_data)
    hourly_data = preprocess_data(hourly_data)
    visualize_data(hourly_data)
    plot_heatmap(hourly_data)
    X, y, X_train, X_test, y_train, y_test = select_features(hourly_data)
    preprocessor = model_pipeline(X)
    trained_models = train_model(preprocessor, X_train, y_train)

    # evaluation on each trained models
    for name, pipe in trained_models.items():
        y_pred = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)

    



if __name__ == '__main__':
    main()