import pandas as pd

from config import REQUIRED_COLS


def validate_data(data: pd.DataFrame) -> None:

    '''
    data validation

    checks:
        - dataset is not empty
        - missing values in dataset columns
    
    args:
        data (pd. DataFrame)

    returns:
        None
    
    '''

    if data.empty:
        raise ValueError(f'dataset is empty')

    for col in REQUIRED_COLS:
        if data[col].isnull().any():
            raise ValueError(f'missing data found in {col}')

        if data[col].astype(str).str.strip().eq(' ').any():
            raise ValueError(f'blank spaces found in {col}')


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    for col in REQUIRED_COLS:
        if col in data:
            data = data.rename(columns={col: col.lower().replace(' ', '_')})
            
        return data