import matplotlib.pyplot as plt
import seaborn as sns

def eda(data):

    print('='*60)
    print('TIME SERIES FORECASTING ON AIR PASSENGERS DATASET')
    print('='*60)
    print(data.head(10))
    print(f'\n ===== SHAPE OF THE DATASET ===== \n {data.shape} ')
    print('===== DATASET INFORMATION ===== \n')
    print(data.info())
    print(f'\n ==== HANDLING MISSING VALUES ===== \n {data.isnull().sum()}' )
    print(f'\n ===== SUMMARY STATISTICS ===== \n {data.describe()}')