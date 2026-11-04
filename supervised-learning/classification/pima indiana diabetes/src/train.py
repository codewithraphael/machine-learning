from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier


def train_models(X_train, y_train):


    models = {
        'LOGISTIC REGRESSION': LogisticRegression(class_weight='balanced', random_state=42),
        'RANDOMFOREST CLASSIFIER': RandomForestClassifier(class_weight='balanced', random_state=42),
        'DECISION TREE CLASSIFIER': DecisionTreeClassifier(random_state=42),
        'K_NEIGHBORS CLASSIFIER': KNeighborsClassifier(random_state=42),
        'XGBOOST CLASSIFIER': XGBClassifier(random_state=42)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models