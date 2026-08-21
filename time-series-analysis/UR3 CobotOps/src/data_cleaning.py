import pandas as pd
from sklearn.impute import SimpleImputer

from config import BINARY_COLUMNS


def clean_data(data, BINARY_COLUMNS=BINARY_COLUMNS):

    '''
    fixing column names trailing space,
    convert Timestamp to Datetime,
    handle missing values,
    drop redundant(Num) column,
    change column names to lowercase
    convert binary columns to integer
    '''

    data = data.copy()

    data.columns = data.columns.str.lower()  # convert column names to lowercase
    data.columns = data.columns.str.strip()  # remove trailing spaces

    if 'num' in data.columns:
        data = data.drop(columns=['num'], axis=1)

    if 'timestamp' in data.columns:
        data['timestamp'] = (
            data['timestamp']
            .astype(str)
            .str.strip()
            .str.replace('"', '', regex=False)
        )

        data['timestamp'] = pd.to_datetime(
            data['timestamp'],
            utc=True,
            errors='coerce'
        )

    data = data.sort_values('timestamp').reset_index(drop=True)

    # renaming robot_protectivestop column name
    if 'robot_protectivestop' in data.columns:
        data = data.rename(columns={'robot_protectivestop': 'protective_stop'})

    data = data.dropna(subset=['protective_stop'])

    excluded_columns = ['timestamp', 'cycle', 'grip_lost', 'protective_stop']
    sensor_columns = [col for col in data.columns if col not in excluded_columns]

    if sensor_columns:
        imputer = SimpleImputer(strategy='median')
        data[sensor_columns] = imputer.fit_transform(data[sensor_columns])


    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
    }

    for col in BINARY_COLUMNS:
        if col in data.columns:
            data[col] = (
                data[col].replace(mapping).astype(int)
            )

    print('='*50)
    print(f'CLEANED DATASET')
    print('='*50)

    print(data.head(n=10))

    return data


def investigate_time(data):

    '''
    most observations are approximately one second apart, but there are some gaps.
    '''
    data = data.copy()

    time_diff = data.index.to_series().diff()

    print('='*50)
    print(time_diff.describe())
    print('='*50)