# %%
# ===== Loading Libraries =====
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, roc_auc_score, roc_curve, RocCurveDisplay


# %%
# ===== Model Evaluation =====
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_report = classification_report(y_test, y_pred)
    

    return accuracy, classification_report


# %%
# ===== Probabilities Prediction =====
def predict_probabilities(model, X_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_prob


