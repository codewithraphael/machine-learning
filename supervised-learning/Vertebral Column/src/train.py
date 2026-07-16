from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier




def train_model(X_train, y_train):
    models = {
        'lr_model': LogisticRegression(class_weight='balanced', random_state=42),
        'svc_model': SVC(probability=True),
        'rfc_model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'xgb_model': XGBClassifier()
    }

    trained_model = {}


    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_model[name] = model

    return trained_model