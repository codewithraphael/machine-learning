from config import TARGET, FUTURE_TARGET, SENSOR_FEATURES

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

