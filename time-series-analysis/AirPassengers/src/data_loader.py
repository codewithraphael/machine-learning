import pandas as pd
from config import filepath


def load_data(file_path):

    file_path = filepath

    if not file_path.exists():
        raise FileNotFoundError(f'File Not Found: {file_path}')
    
    data = pd.read_csv(file_path)

    return data