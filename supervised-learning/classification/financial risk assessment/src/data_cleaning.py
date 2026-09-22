import pandas as pd

def clean_data(data):

    data = data.copy()

    data.columns = data.columns.str.replace('-', '_').str.replace(' ', '_').str.lower()

    print(f'\n ===== CLEANED COLUMN NAMES ===== \n')
    print(data.columns)

    return data