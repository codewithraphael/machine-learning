import joblib
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from config import CAT_COLUMNS

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
    selecting features for regression model, since working with a timeseries based dataset,
    splitting the dataset chronoligically should be used
    '''

    X = hourly_data.drop(columns=['total_count', 'casual', 'registered', 'date', 'rec_id'])
    y = hourly_data['total_count']


    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    

    return X, y, X_train, X_test, y_train, y_test


# ===================================
#  FEATURES ENCODING PIPELINE
# ===================================
def model_pipeline(X, cat_columns=CAT_COLUMNS):

    num_cols = [c for c in X.columns if c not in cat_columns]
    cat_cols = [c for c in cat_columns if c in X.columns]
 
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



# ====================================
#  MODEL EVALUATION & CROSS VALIDATION
# ====================================
def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):

    y_pred = pipe.predict(X_test)
  
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    tss = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(pipe, X_train, y_train, cv=tss, scoring='r2')


    print(f'='*60)
    print(f'\n{name}')
    print(f'='*60)
    print(f'\n ===== TRAINING SCORE ===== \n {train_score:.3f}')
    print(f'\n ===== TEST SCORE ===== \n {test_score:.3f}')
    print(f'\n ===== R2 SCORE ===== \n {r2:3f}')
    print(f'\n ===== MEAN ABSOLUTE ERROR ===== \n {mae:.3f}')
    print(f'\n ===== MEAN SQUARED ERROR ===== \n {mse:.3f}')
    print(f'\n ===== ROOT MEAN SQUARED ERROR ===== \n {rmse:.3f}')
    print(f'\n ===== CROSS-VALIDATION SCORE ===== \n mean: {cv.mean():.3f} standard deviation: (+/- {cv.std() * 2:.3f})')

    metrics = {
        'Train Score': train_score,
        'Test Score': test_score,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'CV Score': cv,
        'CV Mean': cv.mean(),
        'CV Std': cv.std()
    }

    return y_pred, metrics


# ====================================
# MODEL EVALUATION VISUALIZATIONS
# ====================================
def evaluation_plots(predictions, y_test):

    """
    Creates comparison plots for all regression models.

    Parameters
    ----------
    predictions : dict
        Dictionary containing model predictions.
        Example:
        {
            'Linear Regression': y_pred1,
            'RandomForest': y_pred2,
            'XGBoost': y_pred3
        }

    y_test : Series
        True target values.
    """

    # ----------------------------------
    # Actual vs Predicted
    # ----------------------------------
    fig, axes = plt.subplots(1, len(predictions), figsize=(8 * len(predictions), 6))

    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):

        ax.scatter(y_test,
                   y_pred,
                   alpha=0.5)

        # Perfect prediction line
        minimum = min(y_test.min(), y_pred.min())
        maximum = max(y_test.max(), y_pred.max())

        ax.plot([minimum, maximum],
                [minimum, maximum],
                color='red',
                linestyle='--')

        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted_all_models.png")
    plt.close()


    # ----------------------------------
    # Residual Plot
    # ----------------------------------
    fig, axes = plt.subplots(1, len(predictions),
                             figsize=(8 * len(predictions), 6))

    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):

        residuals = y_test - y_pred

        ax.scatter(y_pred,
                   residuals,
                   alpha=0.5)

        ax.axhline(0,
                   color='red',
                   linestyle='--')

        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_plot_all_models.png")
    plt.close()


    # ----------------------------------
    # Prediction Error Distribution
    # ----------------------------------
    fig, axes = plt.subplots(1, len(predictions),
                             figsize=(8 * len(predictions), 6))

    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):

        errors = y_test - y_pred

        ax.hist(errors,
                bins=30)

        ax.set_title(name)
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_error_distribution.png")
    plt.close()


# =========================
# MODEL METRICS COMPARISON
# =========================

def plot_model_metrics(metrics):

    """
    evaluation comparison bar charts for all models.
    """

    metrics_df = pd.DataFrame(metrics).T

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metrics = [
        ("R2", axes[0,0]),
        ("RMSE", axes[0,1]),
        ("MAE", axes[1,0]),
        ("CV Mean", axes[1,1])
    ]

    for metric, ax in metrics:

        values = metrics_df[metric]

        bars = ax.bar(metrics_df.index,
                      values)

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=20)

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "model_metrics_comparison.png",
        dpi=300
    )

    plt.close()


# =========================
#  SAVING BEST MODEL
# =========================
def save_model(trained_models, metrics_results, metric='RMSE'):

    '''
    saving  best-performing model on the given metric
    (lower is better for RMSE/MAE/MSE, higher is better for R2/CV Mean)
    '''

    lower = metric in ('RMSE', 'MAE', 'MSE')

    best_name = min(metrics_results, key=lambda n: metrics_results[n][metric]) if lower \
        else max(metrics_results, key=lambda n: metrics_results[n][metric])

    best_pipe = trained_models[best_name]

    print(f'\n ===== BEST MODEL ({metric}) ===== \n {best_name}')

    joblib.dump(best_pipe, MODELS_DIR / 'best_model.joblib')


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
    preprocessor = model_pipeline(X, cat_columns=CAT_COLUMNS)
    trained_models = train_model(preprocessor, X_train, y_train)

    # evaluation on each trained models
    predictions = {}
    metrics_results ={}

    for name, pipe in trained_models.items():
        y_pred, metrics = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)

        predictions[name] = y_pred
        metrics_results[name] = metrics


    evaluation_plots(predictions, y_test)
    plot_model_metrics(metrics_results)
    save_model(trained_models, metrics_results, metric='RMSE')

   
if __name__ == '__main__':
    main()