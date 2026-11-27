import pandas as pd


def eda(data: pd.DataFrame) -> None:

    '''
    explanatory data analysis

    checks:
        - shape of dataset
        - dataset information
        - missing values
        - duplicate records

    args:
        data (pd.DataFrame)

    returns:
        None
    '''

    print('='*100)
    print(' '*40 + 'HOUSE RENT PREDICTION')
    print('='*100)

    print(data.head(10))

    print(f'\n ==== SHAPE OF DATASET ==== \n {data.shape}')
    print(f'\n ==== DATASET INFORMATION ====  \n {data.info()}')
    print(f'\n ==== MISSING VALUES ==== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ==== DUPLICATE VALUES ==== \n {data.duplicated().sum()}')
    print(f'\n ==== SUMMARY STATISTICS ==== \n {data.describe()}')
