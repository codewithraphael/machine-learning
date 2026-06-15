# %%
# ===== Loading Libraries =====
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold

# %%
# ===== Model Prediction =====
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report


# %%
# ===== Probabilities Prediction =====
def predict_probabilities(model, X_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_prob


# %%
def validate(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_validation = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print(f'\n ===== Cross Validation Scores ===== \n {cross_validation}')
    print(f'\n ===== Mean and Standard Deviation ===== \n {cross_validation.mean():.3f} (+\-) {cross_validation.std() * 2:.3f}')

    return cross_validation