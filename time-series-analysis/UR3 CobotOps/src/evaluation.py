import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_auc_score, roc_curve, classification_report

from config import PLOTS_DIR


def evaluate_models(name, pipe, X_train, X_test, y_train, y_test):

    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tss = TimeSeriesSplit(n_splits=10)
    cv = cross_val_score(pipe, X_train, y_train, cv=tss)

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
        if hasattr(model, 'decision_function'):
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
        print(f'\n ===== Receiver Operating Characteristics Curve Score ===== \n {roc_score:.3f}')

    if feature_importance_df is not None:
        print(f'\n ===== Feature Importances ===== \n {feature_importance_df.to_string(index=False)}')
    

    print(f'\n ===== Cross Validation Score ===== \n {cv}')
    print(f'\n ===== CV Mean & Standard Deviation ===== \n {cv.mean():.3f} (+/-) {cv.std()* 2:.3f}')


    return y_pred, y_prob


def plot_confusion_matrices(trained_models, X_test, y_test):

    num_models = len(trained_models)
    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()

    for idx, (name, pipe) in enumerate (trained_models.items()):

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
    plt.savefig(PLOTS_DIR / 'confusion_matrices_plot.png')
    plt.close()


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
            print(f'could not plot ROC Curve for {name}: {e}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Models Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'roc_curves_plot.png')
    plt.close()

