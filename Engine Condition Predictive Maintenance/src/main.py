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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
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


# =====================================
#  FEATURE SELECTION AND DATA SPLITTING
# =====================================

def feature_selection(data):
    X = data.drop(columns=['Engine Condition'], axis =1)
    y = data['Engine Condition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


# ============================
#  DATA PREPROCESSING PIPELINE
# ============================

def preprocess_data(X):
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

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
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest Classifier': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=2),
        'Xgboost Classifier': XGBClassifier()
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

    return name, trained_models
    

# ====================
# EVALUATE MODEL 
# ====================

def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, X_train, y_train, cv = skf, scoring='roc_auc')
    
    

    # Probabilities, Decision Score, Roc Score, Feature Importance
    y_prob = None
    decision_func = None
    roc_score = None
    feature_imp = None

    try:
        if hasattr(pipe, 'predict_proba'):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            roc_score = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        roc_score = None

    try:
        if hasattr(pipe, 'decision_function'):
            decision_func = pipe.decision_function(X_test)
    except Exception:
        decision_func = None

    try:
        if hasattr(pipe, 'feature_importance_'):
            feature_imp = pipe.feature_importance_
    except Exception:
        feature_imp = None

    
    print(f'='*100)
    print(f'\n{name}')
    print(f'='*100)
    print(f'\n ===== Training Score ===== \n {train_score}')
    print(f'\n ===== Accuracy ===== \n {accuracy}')
    print(f'===== Classification Report ===== \n {report}')
    print(f'\n ===== Confusion Matrix ===== \n {cm}')

    if y_prob is not None:
        print(f' ===== Prediction Probabilities ===== \n {y_prob}')
    
    if decision_func is not None:
        print(f'\n ===== Decision Function ===== \n {decision_func}')
    
    if roc_score is not None:
        print(f'\n ===== Roc Auc Score ===== \n {roc_score}')

    if feature_imp is not None:
        print(f'\n ===== Feature Importance ===== {feature_imp}')

    print(f'\n ===== Cross Validation Score ===== \n {cv}')
    print(f'\n ===== CV Mean & Standard Deviation ===== \n {cv.mean():.3f} (+/-) {cv.std()*2:.3f}')


    return y_pred, y_prob





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

    X, y, X_train, X_test, y_train, y_test = feature_selection(data)
    preprocessor = preprocess_data(X)
    name, trained_models = train_pipeline(preprocessor, X_train, y_train)
    
    # Evaluating for each trained models
    for name, pipe in trained_models.items():
        y_pred, y_prob = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()