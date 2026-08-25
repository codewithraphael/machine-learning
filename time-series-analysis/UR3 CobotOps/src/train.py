from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE


def train_models(preprocessor, X_train, X_test):

    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),


        'randomforest_classifier': RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    }

    trained_models = {}

    for name, model in models.items():

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipe.fit(X_train, X_test)

        trained_models[name] = pipe

    return trained_models