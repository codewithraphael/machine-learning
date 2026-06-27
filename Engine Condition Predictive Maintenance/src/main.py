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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
#  FEATURE ENGINEERING
# ====================

def feature_engineering(data):

    # total pressure
    data['total_pressure'] = (
        data['Lub oil pressure'] 
        + data['Fuel pressure'] 
        + data['Coolant pressure']
    )

    # total temperature
    data['total_temp'] = (
        data['lub oil temp'] 
        + data['Coolant temp']
    )

    # differences
    data['temp_difference'] = (
        data['lub oil temp'] 
        - data['Coolant temp']
    )

    data['pressure_difference'] = (
        data['Lub oil pressure'] 
        - data['Coolant pressure']
    )

    # ratios
    data['temp_ratio'] = (
        data['lub oil temp'] 
        / data['Coolant temp']
    )

    data['pressure_ratio'] = (
        data['Lub oil pressure'] 
        / data['Coolant pressure']
    )

    # rpm interaction
    data['rpm_oil_temp'] = data['Engine rpm'] * data['lub oil temp']
    data['rpm_coolant_temp'] = data['Coolant temp'] * data['Engine rpm']
    data['coolant_temp_per_rpm'] = data['Coolant temp'] / data['Engine rpm']
    data['oil_temp_per_rpm'] = data['lub oil temp'] / data['Engine rpm']


    data['pressure_std'] = data[
        ['Lub oil pressure',
         'Fuel pressure',
         'Coolant pressure']
    ].std(axis=1)

    data['temp_std'] = data[
        ['lub oil temp',
         'Coolant temp']
    ].std(axis=1)

    data['rpm_pressure'] = (
        data['Engine rpm']
        * data['Fuel pressure']
    )

    data['rpm_squared'] = (
        data['Engine rpm'] ** 2
    )

    
    # Engine Stress Index
    data['engine_stress_index'] = ( 
        data['Engine rpm'] *
        data['Fuel pressure'] * 
        data['Coolant temp']
        ) / data['Lub oil pressure'].replace(0, np.nan)

    # load categories
    data["load_level"] = pd.cut(
    data["engine_stress_index"],
    bins=3,
    labels=["Low", "Normal", "High"])

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

    return data


# =====================
#  DISPLAYING NEW DATA
# =====================
def display_new_data(data):
    print(f'\n ===== New Featured Data ===== \n {data}')


# =====================================
#  FEATURE SELECTION AND DATA SPLITTING
# =====================================

def feature_selection(data):
    X = data.drop(columns=['Engine Condition'], axis =1)
    y = data['Engine Condition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X, y, X_train, X_test, y_train, y_test


# ============================
#  DATA PREPROCESSING PIPELINE
# ============================

def preprocess_data(X):
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(), cat_cols)
        ]
    )

    return preprocessor


# ========================
# MODEL TRAINING PIPELINE
# ========================

def train_pipeline(preprocessor, X_train, y_train):
    models = {

    'Logistic Regression': LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=1
    ),

    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=1
    )
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
    

# ====================
# EVALUATE MODEL 
# ====================

def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, X_train, y_train, cv = skf, scoring='roc_auc')
    
    

    # Probabilities, Decision Score, Roc Score, Feature Importance
    y_prob = None
    decision_func = None
    roc_score = None
    feature_importance_df = None

    try:
        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            roc_score = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        roc_score = None

    try:
        model = pipe.named_steps['model']
        if hasattr(model, 'decision_function'):
            decision_func = model.decision_function(X_test)
    except Exception:
        decision_func = None

    try:
        model = pipe.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            preprocessor = pipe.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'Features': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    except Exception:
        feature_importance_df = None

    
    print(f'='*100)
    print(f'\n{name}')
    print(f'='*100)
    print(f'\n ===== Training Score ===== \n {train_score}')
    print(f'\n ===== Accuracy ===== \n {accuracy}')
    print(f'===== Classification Report ===== \n {report}')
    print(f'\n ===== Confusion Matrix ===== \n {cm}')

    if y_prob is not None:
        print(f' ===== Prediction Probabilities ===== \n {y_prob[:10]}')
    
    if decision_func is not None:
        print(f'\n ===== Decision Function ===== \n {decision_func[:10]}')
    
    if roc_score is not None:
        print(f'\n ===== Roc Auc Score ===== \n {roc_score}')

    if feature_importance_df is not None:
        print(f'\n ===== Feature Importance ===== \n {feature_importance_df.to_string(index=False)}')

    print(f'\n ===== Cross Validation Score ===== \n {cv}')
    print(f'\n ===== CV Mean & Standard Deviation ===== \n {cv.mean():.3f} (+/-) {cv.std()*2:.3f}')


    return y_pred, y_prob


# =========================
#  CONFUSION MATRICES PLOT
# =========================

def plot_confusion_matrices(trained_models, X_test, y_test):
    '''
    plot confusion matrices for all trained models
    '''
    num_models = len(trained_models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (name, pipe) in enumerate(trained_models.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='viridis',
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('../plots/confusion_matrix.png', dpi=600, bbox_inches='tight')
    plt.close()


# ====================
#  ROC CURVES PLOTS
# ====================

def plot_roc_curves(trained_models, X_test, y_test):
    '''
    plot ROC Curves for all models
    '''

    plt.figure(figsize=(10, 8))

    for name, pipe in trained_models.items():
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                label=f'{name} (AUC = {roc_auc:.4f})',
                linewidth = 2
            )
        except Exception as e:
            print(f'Could not plot ROC Curve for {name}: {e}')

    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('../plots/roc_curves_all_models.png', dpi=600, bbox_inches='tight')
    plt.close()


# ================================================
# SAVING ALL TRAINED MODELS FOR DEPLOYMENT TESTING
# ================================================

def save_models(trained_models):
    print('\n' + '='*100)
    print('SAVING MODELS')
    print('='*100)

    for name, pipe in trained_models.items():
        model_filename = name.lower().replace(' ', '_').strip() + '.joblib'
        filepath = f'../models/{model_filename}'

        joblib.dump(pipe, filepath)
        print(f'\n Saved {name} to {filepath}')

    print('\n' + '='*100)


# ====================
#  MAIN FUNCTION
# ====================

def main():
    filepath = '../data/engine_data.csv'
    data = load_data(filepath)

    eda(data)
    plot_target(data)
    plot_pairplot(data)
    plot_correlation(data)

    data = feature_engineering(data)
    display_new_data(data)


    X, y, X_train, X_test, y_train, y_test = feature_selection(data)
    preprocessor = preprocess_data(X)
    trained_models = train_pipeline(preprocessor, X_train, y_train)
    
    # Evaluating for each trained models
    for name, pipe in trained_models.items():
        y_pred, y_prob = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)

    # Plot Confusion matrices and ROC Curves for all models
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)

    # Saving all trained models to joblib for deployment testing
    save_models(trained_models)


if __name__ == "__main__":
    main()