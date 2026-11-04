from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


def evaluate_model(X_train, X_test, y_train, y_test, model, feature_names):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

    
    y_proba = None
    decision_function = None
    feature_importances = None
    roc_score = None

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_score = roc_auc_score(y_test, y_proba)

    except Exception as e:
        print(f"Error calculating Predicted Probabilities & ROC AUC: {e}")

    try:
        if hasattr(model, "decision_function"):
            decision_function = model.decision_function(X_test)
    except Exception as e:
        print(f"Error calculating Decision Function: {e}")

    try:
        if hasattr(model, "feature_importances_"):
            feature_importances = zip(feature_names, model.feature_importances_)
    except Exception as e:
        print(f"Error calculating Feature Importances: {e}")



    print('='*150)
    print(' '*60 + f"Model: {model.__class__.__name__}")
    print('='*150)

    print(f"\n ==== Accuracy ==== \n {accuracy:.4f}")
    print(f'\n ==== Classification Report ==== \n {classification_rep}')
    print(f'\n ==== Confusion Matrix ==== \n {conf_matrix}')
    print(f"\n ==== Cross-Validation Accuracy ==== \n {cv.mean():.4f} ± {cv.std():.4f}")


    if y_proba is not None:
        print(f"\n ==== ROC AUC Score ==== \n {roc_score:.4f}")

    if feature_importances is not None:
        print(f"\n ==== Feature Importances ==== \n")
        for feature_name, importance in feature_importances:
            print(f"{feature_name}: {importance:.4f}")

    if decision_function is not None:
        print(f"\n ==== Decision Function ==== \n")
        print(decision_function)
