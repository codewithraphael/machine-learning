from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import TARGET, FUTURE_TARGET, SENSOR_FEATURES, TRAIN_RATIO

def feature_selection(data, sensor_features=SENSOR_FEATURES, future_target=FUTURE_TARGET):
    """
    Selects relevant features for modeling.

    Parameters:
    - data: DataFrame containing the dataset.
    - sensor_features: List of sensor feature column names.
    - future_target: Name of the future target column.

    Returns:
    - List of selected feature column names.
    """

    excluded = {
        'grip_lost',
        'protective_stop',
        'timestamp'
    }

    features = []

    for column in data.columns:
        if column in excluded:
            continue

        if column in sensor_features or '__lag_' in column or '__rolling_' in column:
            features.append(column)

    return features


def split_data(data, features, target=TARGET, train_ratio=TRAIN_RATIO):

    """
    Splits the dataset into training and testing sets.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of feature column names.
    - target: Name of the target column.
    - train_ratio: Proportion of data to be used for training.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """

    if 'timestamp' not in data.columns:
        raise KeyError("'timestamp' column is required for chronological splitting")

    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between 0 and 1')

    ordered_data = (
        data.sort_values('timestamp', kind='stable')
        .reset_index(drop=True)
    )

    
    split_index = int(len(ordered_data) * train_ratio)
    X = ordered_data[features].copy()
    y = ordered_data[target].copy()

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def preprocess_data():

    '''
    preprocessing pipeline that fills missing numerical values
    using the training set median and standardize the predictors.
    '''

    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return preprocessor