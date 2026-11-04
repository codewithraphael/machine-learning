import pandas as pd


def validate_data(data):

    print('='*150)
    print(' '*50 + 'PIMA INDIA DIABETES')
    print('='*150)


    print(data.head(5))
    print(f'\n ==== SHAPE OF THE DATA ==== \n {data.shape}')
    print(f'\n ==== DATA TYPES =====')
    print(f'\n {data.info()}')
    print(f'\n ==== MISSING VALUES ==== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ==== DUPLICATE RECORDS ==== \n {data.duplicated().sum()}')
    print(f'\n ==== SUMMARY STATISTICS ==== \n {data.describe()}')