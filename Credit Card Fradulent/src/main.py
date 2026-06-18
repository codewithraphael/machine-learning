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
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

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

def preprocess_data(X):
    num_cols = X.select_dtypes(include = ['number']).columns.tolist()
    cat_cols = X.select_dtypes(include = ['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers= [
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='drop'
    )

    return preprocessor


# =============================
# BUILDING TRAINING PIPELINE
# =============================
def build_pipeline(preprocessor, X_train, y_train):
    models = {
        'LOGISTIC REGRESSION MODEL': LogisticRegression(class_weight='balanced', random_state=42),
        'RANDOMFOREST CLASSIFIER MODEL': RandomForestClassifier(class_weight='balanced', random_state=42),
        'DECISION TREE CLASSIFIER MODEL': DecisionTreeClassifier(random_state=42),
        'XGBOOST CLASSIFIER MODEL': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
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
    roc_score = None

    try:
        if hasattr(pipe, 'predict_proba'):
                probabilities = pipe.predict_proba(X_test)[:, 1]
                roc_score = roc_auc_score(y_test, probabilities)
    except Exception:
        probabilities = None
        roc_score = None

    try:
        if hasattr(pipe, 'decision_function'):
            decision_scores = pipe.decision_function(X_test)
    except Exception:
        decision_scores = None

    print(f'='*100)
    print(f'\n{name}')
    print(f'='*100)
    print(f'\n  ===== Training Score ===== \n {train_score}')
    print(f'\n ===== {name} Accuracy Score ===== \n  {accuracy}')

    if roc_score is not None:
        print(f'\n ===== Roc Auc Score ===== \n {roc_score}')
    if probabilities is not None:
        print(f'\n ===== {name} Probabilities ===== \n {probabilities}')
    if decision_scores is not None:
        print(f'\n =====  {name} Decision Scores ===== \n {decision_scores}')

    print(f'\n ===== {name} Classification Report ===== \n {report}')
    print(f'\n ===== {name} Confusion Matrix ===== \n {cm}')

    return y_pred, probabilities

# =============================
# CONFUSION MATRICES PLOT
# =============================

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


# =============================
# ROC CURVES PLOTS
# =============================

def plot_roc_curves(trained_models, X_test, y_test):
    '''
    plot ROC Curves for all models
    '''

    plt.figure(figsize=(10, 8))

    for name, pipe in trained_models.items():
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_proba)
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
      y_pred, probabilities = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)

    # Plot Confusion matrices and ROC Curves for all models
    plot_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)


if __name__ == "__main__":
    main()