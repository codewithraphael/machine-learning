import pandas as pd
import numpy as np

def validate_data(data):

    print('='*100)
    print(' '*40 + 'MAGIC Gamma Telescope')
    print('='*100)

    print(data.head(10))
    print(f'\n ==== SHAPE OF THE DATASET ==== \n {data.shape}')
    print(f'\n ==== DATASET INFORMATION ==== \n')
    print(data.info())
    print(f'\n ==== DUPLICATE OBS ==== \n {data.duplicated().sum()}')
    print(f'\n ==== MISSING VALUES ==== \n {data.isnull().sum().sort_values(ascending=False)}')
    print(f'\n ==== SUMMARY STATISTICS ==== \n {data.describe()}')


    print(f'\n ==== TARGET CLASS COUNT ==== ')
    print(f' GAMMA RADIATION : {len(data[data['class'] == 'g'])}') # gamma
    print(f' HADRON RADIATION : {len(data[data['class'] == 'h'])} \n') # hadron