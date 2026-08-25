import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report, ConfusionMatrixDisplay, RocCurveDisplay


def evaluate_models(name, pipe, X_train, X_test, y_train, y_test):

    y_pred = pipe.predict(X_test)
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tss = TimeSeriesSplit(n_splits=10)
    cv = cross_val_score(pipe, X_train, y_train, cv=tss, scoring='roc_auc')

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
        print(f'\n ===== Receiver Operating Characteristics Curve Score ===== \n {roc_score}')

    if feature_importance_df is not None:
        print(f'\n ===== Feature Importances ===== \n {feature_importance_df.to_string(index=False)}')
    

    print(f'\n ===== Cross Validation Score ===== \n {cv}')
    print(f'\n ===== CV Mean & Standard Deviation ===== \n {cv.mean():.3f} (+/-) {cv.std()* 2:.3f}')


    return y_pred, y_prob
