from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

from config import CAT_COLUMNS, NUM_COLUMNS


def feature_selection(data):

    X = data.drop(columns=['risk_rating'], axis=1)
    y = data['risk_rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test, label_encoder


def preprocess_data(X_train, X_test):

    transformer = ColumnTransformer([

        (
            'num', 
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUM_COLUMNS
        ),


        (
            'cat',
            Pipeline([
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), CAT_COLUMNS
        )
    ])

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    return transformer, X_train, X_test