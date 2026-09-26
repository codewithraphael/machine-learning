import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_auc_score, roc_curve, RocCurveDisplay

import joblib

import warnings
warnings.filterwarnings('ignore')

from config import DATA_PATH, MODEL_PATH, PLOTS_PATH, TARGET_COLUMN



def load_data(filepath):

    if not filepath.exists():
        return FileNotFoundError(f'File Not Found: {filepath}')

    data = pd.read_csv(filepath)

    return data


def validate_data(data):

    print('='*160)
    print(' '*60 + 'CERVICAL CANCER BEHAVIOURAL RISK PREDICTION')
    print('='*160)

    print(data.head(5))
    print(f'\n ===== SHAPE OF THE DATASET ===== \n {data.shape}')
    print(f'\n ===== DATASET INFORMATION ===== \n')
    print(data.info())
    print(f'\n ===== MISSING VALUES ===== \n{data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== DUPLICATE COLUMNS ===== \n {data.duplicated().sum()}')
    print(f'\n ===== SUMMARY STATISTICS ===== \n {data.describe()}')


def clean_data(data):

    data = data.copy()

    data.columns = data.columns.str.lower()

    return data


def visualize_data(data):

    sns.pairplot(data)
    plt.savefig(PLOTS_PATH / 'pairplot_distribution.png')
    plt.close()

    plt.figure(figsize=(20, 20))
    sns.heatmap(data.corr(), annot=True, cmap='viridis', linewidths=0.5)
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, linewidth=0.5, cmap='viridis')
    plt.title('Heatmap Correlation Matrix')
    plt.savefig(PLOTS_PATH / 'correlation_matrix.png')
    plt.close()


def preprocess_data():

    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])

    return preprocessor


def feature_selection(data):

    X = data.drop(columns=TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train, preprocessor):

    models = {

        'LOGISTIC REGRESSION': LogisticRegression(class_weight='balanced', random_state=42),
        'RANDOMFOREST CLASSIFIER': RandomForestClassifier(class_weight='balanced', random_state=42),
        'KNEIGHBORS CLASSIFIER': KNeighborsClassifier(),
        'DECSION TREE CLASSIFIER': DecisionTreeClassifier(class_weight='balanced', random_state=42)
    }

    trained_models = {}

    for name, model in models.items():

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)

        trained_models[name] = pipe

    return trained_models 


def evaluate_models(X_train, X_test, y_train, y_test, pipe, name):

    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    train_score = pipe.score(X_train, y_train)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')

    y_proba = None
    roc_score = None
    decision_score = None
    feature_importances = None

    try:
        if hasattr(pipe, 'predict_proba'):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            roc_score = roc_auc_score(y_test, y_proba)
    except Exception as e:
        y_proba = None
        roc_score = None
       
        return AttributeError(f'Attribute Not Found: {e}')

    try:
        if hasattr(pipe, 'decision_function'):
            decision_score = pipe.decision_function(X_test)
    except Exception as e:
        decision_score = None

        return AttributeError(f'Could not compute decision scores: {e}')

    try:
        model = pipe.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
            importance = model.feature_importances_
            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
        elif hasattr(model, 'coef_'):
            feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
            importance = np.abs(model.coef_).mean(axis=0)
            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
        else:
            feature_importances = 'Feature importance not available for this model.'
    except Exception as e:
        feature_importances = None



    print('='*80)
    print(f'\n{name}')
    print('='*80)

    print(f'\n ===== TRAINING SCORE ===== \n {train_score:.3f}')
    print(f'\n ===== TEST SCORE (ACCURACY) ===== \n {accuracy:.3f}')


    if y_proba is not None:
        print(f'\n ===== PREDICTION PROBABILITIES ===== \n {y_proba}')

    if roc_score is not None:
        print(f'\n ===== RECEIVER OPERATING CHARACTERISTICS CURVE SCORE ===== \n {roc_score}')

    if decision_score is not None:
        print(f'\n ===== DECISION SCORES ===== \n {decision_score}')

    print(f'\n ===== CROSS VALIDATION SCORE ===== \n {cv}')
    print(f'\n ===== FEATURE IMPORTANCES ===== \n {feature_importances}')
    print(f'\n ===== CONFUSION MATRIX ===== \n {cm}')
    print(f'\n ===== CLASSIFICATION REPORT ===== \n {report}')

    return y_pred, y_proba

    






def main():

    data = load_data(DATA_PATH)
    validate_data(data)
    data = clean_data(data)
    visualize_data(data)
    preprocessor = preprocess_data()
    X_train, X_test, y_train, y_test = feature_selection(data)
    trained_models = train_models(X_train, y_train, preprocessor)

    for name, pipe in trained_models.items():
        y_pred, y_proba = evaluate_models(X_train, X_test, y_train, y_test, pipe, name)


if __name__ == '__main__':
    main()
