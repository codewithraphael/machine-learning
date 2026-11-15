from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier

from config import RANDOM_STATE


def train_models(X_train, y_train):

    models = {
        'LOGISTIC REGRESSION': LogisticRegression(
            class_weight='balanced', 
            random_state=RANDOM_STATE
        ),
        'RANDOMFOREST CLASSIFIER': RandomForestClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'K-NEIGHBORS CLASSIFIER': KNeighborsClassifier(
            n_neighbors=5
        ),
        'DECISION TREE CLSSIFIER': DecisionTreeClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            splitter='best'
        ),
        'XGBOOST CLASSIFIER': XGBClassifier(
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'MULTILAYER PERCEPTRON': MLPClassifier(
            hidden_layer_sizes= 4,
            random_state=RANDOM_STATE,
            early_stopping=True
        )

    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models