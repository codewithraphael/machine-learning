import pandas as pd

def load_data(filepath):

    if not filepath.exists():
        return FileNotFoundError(f'file not found {filepath}')

    data = pd.read_csv(filepath)

    return data