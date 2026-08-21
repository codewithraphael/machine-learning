import pandas as pd

from pathlib import Path
from config import file_path


def load_data(filepath):

    filepath = Path(file_path)

    if not filepath.exists():
        raise FileNotFoundError(f'File Not Found: {filepath}')       

    data = pd.read_excel(filepath)

    print('='*100)
    print('MULTIVARIATE TIMESERIES ANALYSIS ON UR3COBOTOPS DATASET')
    print('='*100)

    return data