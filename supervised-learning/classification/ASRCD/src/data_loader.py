import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:

    '''
    implementing data loading function

    checks:
        - file exists in path
        - load dataset

    args:
        filepath ( path to file: pd.DataFrame)

    raises:
        FileNotFoundError
            if data not found in path
    '''

    if not filepath.exists():
        raise FileNotFoundError(f'File Not Found in Path')
    
    data = pd.read_csv(filepath)

    return data