import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, auc,  roc_auc_score, roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

def evaluate(name, pipe, X_train, X_test, y_train, y_test):
    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')


    y_prob = None
    decision_scores = None
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
        if hasattr(model, 'decision function'):
            decision_scores = model.decision_function(X_test)
    except Exception:
        decision_scores = None

    try:
        model = pipe.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            preprocessor = pipe.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'Features': feature_names,
                'Importances': importances
            }).sort_values(by='Importances', ascending=False).reset_index(drop=True)
    except Exception:
        feature_importance_df = None

    

    print(f'='*120)
    print(f'\n{name}')
    print(f'='*120)

    print(f'\n ===== Training Score ===== \n {train_score:.3f}')
    print(f'\n ===== Test Score ===== \n {test_score:.3f}')
    print(f'\n ===== Accuracy Score ===== \n {accuracy:.3f}')
    print(f'\n ===== Classification Report ===== \n {report}')
    print(f'\n ===== Confusion Matrix ===== \n {cm}')

    if y_prob is not None:
        print(f'\n ===== Prediction Probabilities ===== \n {y_prob[:10]}')

    if decision_scores is not None:
        print(f'\n ===== Decision Function Scores ===== \n {decision_scores[:10]}')
    
    if roc_score is not None:
        print(f'\n ===== Receiver Operating Characteristics Curve Score ===== \n {roc_score}')

    if feature_importance_df is not None:
        print(f'\n ===== Feature Importances ===== \n {feature_importance_df.to_string(index=False)}')
    

    print(f'\n ===== Cross Validation Score ===== \n {cv}')
    print(f'\n ===== CV Mean & Standard Deviation ===== \n {cv.mean():.3f} (+/-) {cv.std()* 2:.3f}')


    return y_pred, y_prob


def plot_confusion_matrices(trained_models, X_test, y_test):

    num_models = len(trained_models)
    fig, axes = plt.subplots(1, 5)
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



def plot_roc_curves(trained_models, X_test, y_test):
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
    plt.savefig('../plots/roc_curves_plot.png', dpi=600, bbox_inches='tight')
    plt.close()