import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:

    if not filepath.exists():
        raise FileNotFoundError(f'File Not Found in Path')
    
    data = pd.read_csv(filepath)

    return data