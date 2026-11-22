from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(pipe, x_train, x_test, y_train, y_test):

    y_pred = pipe.predict(x_test)
    train_score = pipe.score(x_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print('='*100)
    print(' '*30 + 'TFIDF-VECTORIZER / LOGISTIC REGRESSION')
    print('='*100)
    
    print(f'\n ==== TRAIN SCORE ==== \n {train_score:.3f}')
    print(f'\n ==== ACCURACY SCORE ==== \n {accuracy:.3f}')
    print(f'\n ==== CONFUSION MATRIX ==== \n {cm}')
    print(f'\n ==== CLASSIFICATION REPORT ==== \n {report}')