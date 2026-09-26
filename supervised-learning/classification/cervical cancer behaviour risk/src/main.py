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

from config import DATA_PATH, MODEL_PATH, PLOTS_PATH, EVALUATION_PATH, TARGET_COLUMN



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
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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
    except Exception:
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
    print(f'\n ===== CLASSIFICATION REPORT ===== \n {report_text}')


    metrics = {
        'Model': name,
        'Train Score': float(train_score),
        'Test Accuracy': float(accuracy),
        'ROC AUC': float(roc_score) if roc_score is not None else np.nan,
        'CV Mean Score': float(cv.mean()) if len(cv) else np.nan,
        'CV Scores': ', '.join(f'{score:.4f}' for score in cv),
        'Precision_0': float(report_dict.get('0', {}).get('precision', 0)),
        'Recall_0': float(report_dict.get('0', {}).get('recall', 0)),
        'F1_0': float(report_dict.get('0', {}).get('f1-score', 0)),
        'Support_0': int(report_dict.get('0', {}).get('support', 0)),
        'Precision_1': float(report_dict.get('1', {}).get('precision', 0)),
        'Recall_1': float(report_dict.get('1', {}).get('recall', 0)),
        'F1_1': float(report_dict.get('1', {}).get('f1-score', 0)),
        'Support_1': int(report_dict.get('1', {}).get('support', 0)),
        'Macro Avg Precision': float(report_dict.get('macro avg', {}).get('precision', 0)),
        'Macro Avg Recall': float(report_dict.get('macro avg', {}).get('recall', 0)),
        'Macro Avg F1': float(report_dict.get('macro avg', {}).get('f1-score', 0)),
        'Weighted Avg Precision': float(report_dict.get('weighted avg', {}).get('precision', 0)),
        'Weighted Avg Recall': float(report_dict.get('weighted avg', {}).get('recall', 0)),
        'Weighted Avg F1': float(report_dict.get('weighted avg', {}).get('f1-score', 0)),
        'Confusion Matrix': str(cm.tolist())
    }

    return y_pred, y_proba, metrics



def save_model_comparison_csv(model_metrics, export_path):

    export_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.to_csv(export_path, index=False)

    return metrics_df


def plot_confusion_matrices(y_test, predictions):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (name, y_pred) in enumerate(predictions.items()):

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        sns.heatmap(
            cm,
            annot=True,
            cmap='viridis',
            fmt='d',
            ax=axes[idx],
            cbar=False
        )
        axes[idx].set_title(f'{name}\nAccuracy; {accuracy_score(y_test, y_pred):.4f}')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / 'confusion_matrices_plot.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(y_test, probabilities):

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f'{name} (AUC = {roc_auc:.4f})'
        )

    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Models Performance Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / 'roc_curve_plots.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


def save_model(model, model_name):

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    model_name = model_name.replace(" ", "_").lower()
    model_file_path = MODEL_PATH / f'{model_name}.joblib'
    joblib.dump(model, model_file_path)






def main():

    data = load_data(DATA_PATH)
    validate_data(data)
    data = clean_data(data)
    visualize_data(data)
    preprocessor = preprocess_data()
    X_train, X_test, y_train, y_test = feature_selection(data)
    trained_models = train_models(X_train, y_train, preprocessor)

    predictions = {}
    probabilities = {}
    model_metrics = []

    for name, pipe in trained_models.items():
        y_pred, y_proba, metrics = evaluate_models(X_train, X_test, y_train, y_test, pipe, name)
        predictions[name] = y_pred
        probabilities[name] = y_proba
        model_metrics.append(metrics)

    save_model_comparison_csv(model_metrics, EVALUATION_PATH)
    plot_confusion_matrices(y_test, predictions)
    plot_roc_curves(y_test, probabilities)

    save_model(trained_models['LOGISTIC REGRESSION'], 'logistic_regression')


if __name__ == '__main__':
    main()
