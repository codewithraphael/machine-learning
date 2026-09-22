from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

def train_model(X_train, y_train):

    models = {
        'LOGISTIC REGRESSION': LogisticRegression(max_iter=1000),
        'DECISION TREE': DecisionTreeClassifier(random_state=42),
        'RANDOM FOREST': RandomForestClassifier(random_state=42),
        'XGBOOST': XGBClassifier(random_state=42)
    }

    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models