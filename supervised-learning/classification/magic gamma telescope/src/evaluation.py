from sklearn.model_selection import StratifiedKFold, cross_val_score, ValidationCurveDisplay, validation_curve
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix

from config import FEATURE_COLS, RANDOM_STATE


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
                y_proba = model.predict_proba(X_test)[:1]
                roc_score = roc_auc_score(y_test, y_proba)
        except Exception as e:
            print(f'Error evaluating predicted probabilities & ROC AUC: {e}')

        try:
            if hasattr(model, 'decision_function'):
                decision_func = model.decision_function(X_test)
        except Exception as e:
            print(f'Error evaluating Decision Function: {e}')

        try:
            if hasattr(model, 'features_importances_'):
                feature_importances = zip(features_names, model.feature_importances_)
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
            print(f"\n ==== ROC AUC Score ==== \n {roc_score}")

        if feature_importances is not None:
            print(f"\n ==== Feature Importances ==== \n")
            for feature_name, importance in feature_importances:
                print(f"{feature_name}: {importance:.4f}")

        if decision_func is not None:
            print(f"\n ==== Decision Function ==== \n")
            print(decision_func)

    

