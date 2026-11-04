from config import FEARTURE_COLUMNS, TARGET_COLUMN
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def feature_selection(data):
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), FEARTURE_COLUMNS),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return preprocessor, X_train, X_test
