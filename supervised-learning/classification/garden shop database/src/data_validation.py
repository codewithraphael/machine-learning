import pandas as pd


def validate_query(query):

    print('='*100)
    print(' '*40 + 'GARDEN SHOP DATABASE')
    print('='*100)


    print(query)
    print(f'\n ===== SHAPE OF THE DATASET ===== \n {query.shape}')
    print(f'\n ===== DATASET INFORMATION ===== \n')
    print(query.info())
    print(f'\n ===== MISSING VALUES ===== \n {query.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ===== DUPLICATE VALUES ===== \n {query.duplicated().sum()}')
    print(f'\n ===== SUMMARY STATISTICS ===== \n {query.describe()}')
