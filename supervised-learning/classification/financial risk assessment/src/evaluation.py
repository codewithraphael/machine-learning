import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)

from config import PLOTS_PATH

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average='macro',
        zero_division=0
    )
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc_ovr')


    y_proba = None
    decision_scores = None
    roc_score = None
    feature_importance_df = None

    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            roc_score = roc_auc_score(
                y_test,
                y_proba,
                multi_class='ovr',
                average='weighted'
            )
    except Exception as e:
        print(f'Error occurred while calculating ROC-AUC: {e}')
        y_proba = None
        roc_score = None

    try:
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X_test)
    except Exception as e:
        print(f'Error occurred while calculating decision scores: {e}')
        decision_scores = None

    try:
        if hasattr(model, 'feature_importances_'):
            feature_names = getattr(
                X_train,
                'columns',
                [f'feature_{idx}' for idx in range(X_train.shape[1])]
            )
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
    except Exception as e:
        print(f'Error occurred while evaluating model: {e}')
        feature_importance_df = None



    print(f'='*120)
    print(f'\n{name}')
    print(f'='*120)

    print(f'\n ===== Training Score ===== \n {train_score:.3f}')
    print(f'\n ===== Test Score ===== \n {test_score:.3f}')
    print(f'\n ===== Accuracy Score ===== \n {accuracy:.3f}')
    print(f'\n ===== Classification Report ===== \n {report}')
    print(f'\n ===== Confusion Matrix ===== \n {cm}')
    print(f'\n ===== Cross-Validation ROC-AUC (OvR) ===== \n {cv.mean():.3f}')

    if y_proba is not None:
        print(f'\n ===== Prediction Probabilities ===== \n {y_proba[:5]}')

    if decision_scores is not None:
        print(f'\n ===== Decision Function Scores ===== \n {decision_scores[:5]}')
    
    if roc_score is not None:
        print(f'\n ===== Receiver Operating Characteristics Curve Score ===== \n {roc_score:.3f}')

    if feature_importance_df is not None:
        print(f'\n ===== Feature Importances ===== \n {feature_importance_df.head(10).to_string(index=False)}')

    return {
        'Model': name,
        'Accuracy': accuracy,
        'Balanced Accuracy': balanced_accuracy,
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1': f1,
        'CV ROC-AUC': cv.mean()
    }


def plot_model_comparison(trained_models, X_test, y_test):
    if not trained_models:
        return

    comparison = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average='macro',
            zero_division=0
        )
        comparison.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Macro Precision': precision,
            'Macro Recall': recall,
            'Macro F1': f1
        })

    comparison_df = pd.DataFrame(comparison).set_index('Model')
    ax = comparison_df.plot(
        kind='bar',
        figsize=(13, 7),
        width=0.8,
        ylim=(0, 1),
        rot=0,
        color=['#4472C4', '#70AD47', '#ED7D31', '#A5A5A5', '#FFC000']
    )
    ax.set_title('Multiclass Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / 'model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)