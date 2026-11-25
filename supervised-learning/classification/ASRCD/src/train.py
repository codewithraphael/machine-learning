import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

from config import TEXT_COLUMN, TEST_SIZE, RANDOM_STATE

def split_features(data: pd.DataFrame, target):

    X = data[TEXT_COLUMN].astype(str)
    y = data[target].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, pipe, tune=False):

    '''
    training model on one target each and hyperparameter tuning
    '''
    best_params = None

    if tune:
        class_count = y_train.value_counts().min()
        if class_count >= 2:
            cv = min(5, int(class_count))
            search = GridSearchCV(
                estimator= pipe,
                param_grid={
                    'classifier__C': [
                        0.1,
                        1.0,
                        3.0,
                        10.0
                    ]
                },
                scoring = 'f1_macro',
                cv=cv,
                n_jobs=1,
                refit=True
            )

            search.fit(x_train, y_train)
            pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            pipe.fit(x_train, y_train)
    else:
        pipe.fit(x_train, y_train)

    return pipe, best_params