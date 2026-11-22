import pandas as pd

from config import REQUIRED_COLUMNS

def validate_data(data: pd.DataFrame) -> None:

    '''
    validate dataset
    
    checks:
        - dataset is not empty
        - required input and target columns contains no missing values
        - required input and target columns contains no blank strings
    
    args:
        data (pd.Dataframe):
            dataset to validate

    raise:
        ValueError
            if validation fails
    '''

    
    if data.empty:
        raise ValueError('Dataset is Empty')


    for col in REQUIRED_COLUMNS:
        if data[col].isnull().any():
            raise ValueError(f'missing values found in {col}')

        if data[col].astype(str).str.strip().eq("").any():
            raise ValueError(f'blank spaces found in {col}')