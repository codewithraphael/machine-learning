from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import StratifiedKFold, cross_val_score, ValidationCurveDisplay
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, classification_report, confusion_matrix

from config import FEATURE_COLS, RANDOM_STATE, PLOTS_DIR


def evaluate_model(X_test, X_val, y_test, y_val, model, name, features_names=FEATURE_COLS):


        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv = cross_val_score(model, X_val, y_val, cv=skf, scoring='roc_auc')

        y_proba = None
        decision_func = None
        feature_importances = None
        roc_score = None

        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_score = roc_auc_score(y_test, y_proba)
        except Exception as e:
            print(f'Error evaluating predicted probabilities & ROC AUC: {e}')

        try:
            if hasattr(model, 'decision_function'):
                decision_func = model.decision_function(X_test)
        except Exception as e:
            print(f'Error evaluating Decision Function: {e}')

        try:
            estimator = getattr(model, 'named_steps', {}).get('classifier', model)
            if hasattr(estimator, 'feature_importances_'):
                feature_importances = zip(features_names, estimator.feature_importances_)
        except Exception as e:
            print(f'Error evaluating Feature Importances: {e}')


        print('='*150)
        print(' '*60 + f"Model: {name}")
        print('='*150)

        print(f"\n ==== Accuracy ==== \n {accuracy:.4f}")
        print(f'\n ==== Classification Report ==== \n {report}')
        print(f'\n ==== Confusion Matrix ==== \n {cm}')
        print(f"\n ==== Cross-Validation Accuracy ==== \n {cv.mean():.4f} ± {cv.std():.4f}")


        if y_proba is not None:
            print(f"\n ==== ROC AUC Score ==== \n {roc_score:.4f}")

        if feature_importances is not None:
            print(f"\n ==== Feature Importances ==== \n")
            for feature_name, importance in feature_importances:
                print(f"{feature_name}: {importance:.4f}")
        else:
            print("\n Feature Importances: Not available for this model.")

        if decision_func is not None:
            print(f"\n ==== Decision Function ==== \n")
            print(decision_func)

    

def plot_confusion_matrices(trained_models, X_test, y_test):

    num_models = len(trained_models)
    fig, axes = plt.subplots(2, 3)
    axes = axes.ravel()

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
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
    plt.savefig(PLOTS_DIR / f'confusion_matrices_plot.png', dpi=600, bbox_inches='tight')
    plt.close()



def plot_roc_curves(trained_models, X_test, y_test):
    plt.figure(figsize=(10, 8))

    for name, model in trained_models.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]

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
    plt.savefig(PLOTS_DIR / 'receiver_operating_characteristic_curves.png', dpi=600, bbox_inches='tight')
    plt.close()


def plot_validation_curves(trained_models, X_train, y_train):
    parameter_ranges = {
        'LOGISTIC REGRESSION': ('classifier__C', [0.1, 1, 10]),
        'RANDOMFOREST CLASSIFIER': ('classifier__max_depth', [5, 10, 20]),
        'K-NEIGHBORS CLASSIFIER': ('classifier__n_neighbors', [3, 5, 9]),
        'DECISION TREE CLSSIFIER': ('classifier__max_depth', [5, 10, 20]),
        'XGBOOST CLASSIFIER': ('classifier__max_depth', [3, 5, 7]),
        'MULTILAYER PERCEPTRON': ('classifier__alpha', [0.0001, 0.01, 1]),
    }

    if not trained_models:
        raise ValueError('trained_models must contain at least one model')

    unsupported_models = set(trained_models) - set(parameter_ranges)
    if unsupported_models:
        raise ValueError(f'No validation-curve parameter range is configured for: {sorted(unsupported_models)}')

    n_columns = min(3, len(trained_models))
    n_rows = ceil(len(trained_models) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(6 * n_columns, 4.5 * n_rows), squeeze=False)
    axes = axes.ravel()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for ax, (name, model) in zip(axes, trained_models.items()):
        param_name, param_range = parameter_ranges[name]
        ValidationCurveDisplay.from_estimator(
            model,
            X_train,
            y_train,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            scoring='roc_auc',
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel(param_name.split('__')[-1])
        ax.set_ylabel('ROC-AUC')
        ax.grid(alpha=0.3)

    for ax in axes[len(trained_models):]:
        ax.set_visible(False)

    fig.suptitle('Validation Curves by Model (5-fold ROC-AUC)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / 'validation_curves_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)