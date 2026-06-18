# =============================
# IMPORTING LIBRARIES
# =============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix

from xgboost import XGBClassifier

import joblib

import warnings
warnings.filterwarnings('ignore')


# =============================
# LOADING DATASET
# =============================

def load_data(filepath):
    data = pd.read_csv(filepath)

    return data


# =============================
# EXPLANATORY DATA ANALYSIS
# =============================

def eda(data):
    shape = data.shape
    info = data.info()
    missing_value = data.isnull().sum().sort_values(ascending= False)
    statistics = data.describe()

    print(f'\n ===== Shape of the dataset ===== \n {shape}')
    print(info)
    print(f'\n ===== Missing Values ===== \n {missing_value}')
    print(f'\n ===== Summary Statistics ===== \n {statistics}')


# =============================
# COUNTPLOTS
# =============================

def count_plot(data):
    sns.countplot(data, x = 'is_fraud')
    plt.title('Target Distribution')
    plt.savefig('../plots/target_distribution.png')
    plt.close()


# =============================
# CORRELATION HEATMAP
# =============================

def correlation_plot(data):
    corr_plot = data.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_plot, annot=True, linewidths=0.5, cmap='viridis')
    plt.title('Features Correlation Heatmap')
    plt.savefig('../plots/correlation_heatmap.png')
    plt.close()


# =====================================
# FEATURES SELECTION AND DATA SPLITTING
# =====================================

def feature_selection(data):
    X = data.drop(['is_fraud'], axis =1)
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X, y, X_train, X_test, y_train, y_test

# =============================
#  DATA PREPROCESSING PIPELINE 
# =============================

def preprocess_data(data):
    num_cols = data.select_dtypes(include = ['number']).columns.tolist()
    cat_cols = data.select_dtypes(include = ['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers= [
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
        ], remainder='drop'
    )

    return preprocessor


# =============================
# BUILDING TRAINING PIPELINE
# =============================
def build_pipeline(preprocessor, X_train, y_train):
    models = {
        'lr_model': LogisticRegression(class_weight='balanced', random_state=42),
        'rfc_model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'dtc_model': DecisionTreeClassifier(random_state=42),
        'xgb_model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps= [
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        pipe.fit(X_train, y_train)

        trained_models[name] = pipe

    return trained_models



# =============================
# MODEL EVALUATION
# =============================

def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Probabilities / Decision Scores
    probabilities = None
    decision_scores = None
    roc_auc = None

    try:
        if hasattr(pipe, 'predict_proba'):
                probabilities = pipe.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, probabilities)
    except Exception:
        probabilities = None

    try:
        if hasattr(pipe, 'decision_function'):
            decision_scores = pipe.decision_function(X_test)
    except Exception:
        decision_scores = None

    print(f'\n{name}')
    print(f'\n  ===== Training Score ===== \n {train_score}')
    print(f'\n ===== {name} Accuracy Score ===== \n  {accuracy}')

    if roc_auc is not None:
        print(f'\n ===== {name} Probabilities ===== \n {probabilities}')
    if decision_scores is not None:
        print(f'\n ===== Decision Scores ===== \n {decision_scores}')

    print(f'\n ===== {name} Classification Report ===== \n {report}')
    print(f'\n ===== {name} Confusion Matrix ===== \n {cm}')


# =============================
# MAIN FUNCTION
# =============================

def main():
    filepath = '../data/credit_card_fraud.csv'
    data = load_data(filepath)
    data = data.drop(columns=['transaction_id'], errors='ignore')
    
    eda(data)
    count_plot(data)
    correlation_plot(data)
    
    X, y, X_train, X_test, y_train, y_test = feature_selection(data)
    preprocessor = preprocess_data(X)
    trained_models = build_pipeline(preprocessor, X_train, y_train)
    
    # Evaluating on each trained pipeline
    for name, pipe in trained_models.items():
        evaluate_model(name, pipe, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()