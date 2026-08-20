import pandas as pd

from sklearn.impute import SimpleImputer


def clean_data(data):

    '''
    fixing column names trailing space,
    convert Timestamp to Datetime,
    handle missing values,
    drop redundant(Num) column,
    change column names to lowercase
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

    # rename the column without using inplace=True, which returns None and breaks the pipeline
    if 'robot_protectivestop' in data.columns:
        data = data.rename(columns={'robot_protectivestop': 'protective_stop'})

    data = data.dropna(subset=['protective_stop'])

    excluded_columns = ['timestamp', 'cycle', 'grip_lost', 'protective_stop']
    sensor_columns = [col for col in data.columns if col not in excluded_columns]

    if sensor_columns:
        imputer = SimpleImputer(strategy='median')
        data[sensor_columns] = imputer.fit_transform(data[sensor_columns])

    return data