import pandas as pd


def validate_data(data):

    print('='*150)
    print(' '*60 + 'FINANCIAL RISK ASSESSMENT')
    print('='*150)


    print(f'\n {data.head(5)}')
    print(f'\n ===== SHAPE OF THE DATASET ===== \n {data.shape}')
    print(f'\n ===== DATASET INFORMATION ===== \n')
    print(data.info())
    print(f'\n ===== MISSING VALUES ===== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== DUPLICATE COLUMNS ===== \n {data.duplicated().sum()}')
    print(f'\n ===== SUMMARY STATISTICS ==== \n {data.describe()}')