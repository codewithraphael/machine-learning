import pandas as pd

from config import NAMES, COL_NAMES

def load_data(filepath):

    try:

        with open(NAMES, 'r') as file:
            description = file.read()
            
        print(description)


        data = pd.read_csv(filepath, header = None, names=COL_NAMES)

        return data
    except FileNotFoundError:
        print(f' Error File {filepath} was not found')