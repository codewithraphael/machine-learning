from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import TARGET_COL, TEST_SIZE, VAL_SIZE, RANDOM_STATE


def split_data(data, test_size=TEST_SIZE, validation_size=VAL_SIZE, random_state=RANDOM_STATE):


    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError('test_size and validation_size must be positive and sum to less than 1')

    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    relative_validation_size = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train, X_val, X_test):
    preprocessor = Pipeline([
        ('scaler', StandardScaler()),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return X_train, X_val, X_test, preprocessor