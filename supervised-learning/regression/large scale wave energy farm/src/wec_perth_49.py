import joblib
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_theme()

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / 'data/WEC/WEC_Perth_49.csv'
PLOTS_PATH = PROJECT_DIR / 'plots'
MODELS_PATH = PROJECT_DIR / 'models'
EVALUATION_PATH = PROJECT_DIR / 'evaluation_results'


# ====================================
#  LOADING DATASET
# ====================================

def load_data(filepath):
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f'File not found: {filepath}')

    return pd.read_csv(filepath)


# ====================================
#  DATA VALIDATION
# ==================================== 

def validate_data(data):

    print('='*100)
    print(' '*50 + 'WEC PERTH 49 ANALYSIS')
    print('='*100)

    print(data.head(5))

    print(f'\n ===== SHAPE OF DATASET ===== \n {data.shape}')
    print(f'\n ===== DATA TYPES ===== \n {data.dtypes}')
    print(f'\n ===== MISSING VALUES ===== \n {data.isnull().sum().sort_values(ascending=False)}')
    print('\n ===== DATA INFO =====')
    data.info()
    print(f'\n ===== DUPLICATE VALUES ===== \n {data.duplicated().sum()}')
    print(f'\n ===== UNIQUE VALUES ===== \n {data.nunique()}')
    print(f'\n ===== SUMMARY STATISTICS ===== \n {data.describe()}')


# ====================================
#  DATA CLEANING
# ====================================

def clean_data(data):
    data = data.drop_duplicates().dropna().copy()
    return data


# ====================================
# DATA VISUALIZATION 
# ====================================

def visualize_data(data):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.histplot(data['qW'], bins=30, kde=True)
    plt.title('Reactive Power Distribution')
    plt.xlabel('Reactive Power (kVAR)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_PATH / 'wec_perth_49_reactive_power_distribution.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(data['Total_Power'], bins=30, kde=True)
    plt.title('Total Power Distribution')
    plt.xlabel('Total Power (kW)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_PATH / 'wec_perth_49_total_power_distribution.png', dpi=150)
    plt.close()


# ==========================================
#  DATA PREPROCESSING AND FEATURE SELECTION
# ==========================================

def preprocess_data(data):
    features = data.drop(columns=['Total_Power'])
    target = data['Total_Power']

    X = features
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


# ====================================
#  MODEL TRAINING
# ====================================

def train_model(X_train, y_train):
    models = {
        'LINEAR REGRESSION': LinearRegression(),
        'RANDOM FOREST': RandomForestRegressor(
            n_estimators=20,
            random_state=42,
            n_jobs=-1
        ),
        'XGBOOST': XGBRegressor(
            n_estimators=20,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1,
            tree_method='hist'
        )
    }

    trained_models = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return trained_models


# ====================================
#  MODEL EVALUATION
# ====================================

def evaluate_model(X_train, X_test, y_train, y_test, name, model):
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

    metrics = {
        'MODEL': name,
        'TRAIN Score': train_score,
        'TEST Score': test_score,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'CV Mean R2': cv.mean(),
        'CV Std': cv.std()
    }

    return metrics


# ====================================
#  MODEL COMPARISON PLOTS
# ====================================

def plot_model_comparison(trained_models, X_test, y_test, results_df):
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    model_predictions = {
        name: model.predict(X_test)
        for name, model in trained_models.items()
    }

    figure, axes = plt.subplots(1, len(model_predictions), figsize=(18, 5), squeeze=False)
    for axis, (name, predictions) in zip(axes[0], model_predictions.items()):
        axis.scatter(y_test, predictions, alpha=0.6, edgecolor='none')
        minimum = min(y_test.min(), predictions.min())
        maximum = max(y_test.max(), predictions.max())
        axis.plot([minimum, maximum], [minimum, maximum], color='red', linestyle='--')
        axis.set_title(name.title())
        axis.set_xlabel('Actual Total Power (kW)')
        axis.set_ylabel('Predicted Total Power (kW)')
        axis.grid(alpha=0.25)

    figure.suptitle('Actual vs Predicted Total Power')
    figure.tight_layout()
    figure.savefig(PLOTS_PATH / 'model_actual_vs_predicted_comparison.png', dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(model_predictions), figsize=(18, 5), squeeze=False)
    for axis, (name, predictions) in zip(axes[0], model_predictions.items()):
        residuals = y_test - predictions
        axis.scatter(predictions, residuals, alpha=0.6, edgecolor='none')
        axis.axhline(0, color='red', linestyle='--')
        axis.set_title(name.title())
        axis.set_xlabel('Predicted Total Power (kW)')
        axis.set_ylabel('Residual (kW)')
        axis.grid(alpha=0.25)

    figure.suptitle('Residual Comparison')
    figure.tight_layout()
    figure.savefig(PLOTS_PATH / 'model_residual_comparison.png', dpi=150)
    plt.close(figure)

    metric_columns = {'R2': 'R2', 'MAE': 'MAE', 'RMSE': 'RMSE'}
    figure, axes = plt.subplots(1, len(metric_columns), figsize=(15, 5), squeeze=False)
    for axis, (column, title) in zip(axes[0], metric_columns.items()):
        sns.barplot(data=results_df, x='MODEL', y=column, ax=axis, color='steelblue')
        axis.set_title(title)
        axis.set_xlabel('')
        axis.set_ylabel(title)
        axis.tick_params(axis='x', rotation=25)
        axis.grid(axis='y', alpha=0.25)

    figure.suptitle('Model Performance Comparison')
    figure.tight_layout()
    figure.savefig(PLOTS_PATH / 'model_metrics_comparison.png', dpi=150)
    plt.close(figure)


# ====================================
#  MODEL SAVING
# ====================================

def save_best_model(best_name, best_model):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    filename = MODELS_PATH / f'{best_name.lower().replace(" ", "_")}_model.joblib'

    with open(filename, 'wb') as f:
        joblib.dump(best_model, f)

    print(f'\n ===== BEST MODEL SAVED ===== \n {filename}')


# ====================================
#  MAIN
# ====================================

def main():
    EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    data = load_data(DATA_PATH)
    validate_data(data)
    data = clean_data(data)
    visualize_data(data)

    X, y, X_train, X_test, y_train, y_test = preprocess_data(data)
    trained_models = train_model(X_train, y_train)

    results = []

    for name, model in trained_models.items():
        metrics = evaluate_model(X_train, X_test, y_train, y_test, name, model)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    plot_model_comparison(trained_models, X_test, y_test, results_df)
    print(f'\n ===== MODEL COMPARISON =====')
    print(results_df[['MODEL', 'R2', 'MAE', 'RMSE', 'CV Mean R2']].to_string(index=False))

    best_name = results_df.iloc[0]['MODEL']
    best_model = trained_models[best_name]
    save_best_model(best_name, best_model)

    results_df.to_csv(EVALUATION_PATH / 'model_comparison.csv', index=False)
    print(f'\n ===== BEST MODEL ===== \n {best_name}')


if __name__ == '__main__':
    main()