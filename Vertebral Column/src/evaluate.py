from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f'\n ===== Predictions ===== \n {y_pred}')
    print(f'\n ===== Training Score ===== \n {train_score:.3f}')
    print(f'\n ===== Accuracy Score ===== \n {accuracy:.3f}')
    print(f'\n ===== Classification Report ===== \n {report}')
    print(f'\n ===== Confusion Matrix ===== \n {cm}')

    # Try to obtain decision scores / probabilities for ROC AUC
    y_score = None
    if hasattr(model, 'predict_proba'):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
            print(f'\n ===== Prediction Probabilities (positive class) ===== \n {y_score}')
        except Exception:
            y_score = None
    elif hasattr(model, 'decision_function'):
        try:
            y_score = model.decision_function(X_test)
            print(f'\n ===== Decision function scores ===== \n {y_score}')
        except Exception:
            y_score = None

    return y_score


def roc_score(y_test, y_score):
    if y_score is None:
        print('\n ===== Roc Auc Score =====: score unavailable for this model (no predict_proba/decision_function)')
        return None

    ras = roc_auc_score(y_test, y_score)
    print(f'\n ===== Roc Auc Score ===== {ras:.3f}')
    return ras


def cross_validate(trained_model, X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in trained_model.items():
        cv = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
        print(f'\n ==== Cross Validation Score ({name}) ==== \n {cv}')
        print(f'\n ==== Mean / Standard Deviation ({name}) ==== \n {cv.mean():.3f} (+/-) {cv.std() * 2:.3f}')