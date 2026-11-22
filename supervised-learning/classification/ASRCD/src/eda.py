import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from config import PLOTS_DIR, TEXT_COLUMN

def eda(data: pd.DataFrame) -> None:

    '''
    exploratory data analysis and basic statistical exploration

    checks:
        - inspect data
        - analyze target distribution
        - visualize target distribution

    args:
        data (pd.DataFrame)

    ''' 

    print('='*100)
    print(' '* 25 + 'AI SAFETY RESPONSE CLASSIFICATION EVALUATING LLM')
    print('='* 100)

    print(f'\n ==== SHAPE OF THE DATASET ==== \n {data.shape}')
    print(data.head(10))
    print(f'\n ==== DATASET INFORMATION ==== \n')
    print(data.info())
    print(f'\n ==== MISSING VALUES ==== \n {data.isnull().sum()}')
    print(f'\n ==== DUPLICATE RECORDS ==== \n {data.duplicated.sum()}')
    print(f'\n ==== DUPLICATE QUERIES ==== \n {data[TEXT_COLUMN].duplicated.sum()}')


def visualize_target_distribution(data: pd.DataFrame) -> None:

    '''
    generate bar chart distribution showing the distribution of each target variable

    args:
        data (pd.DataFrame)

    path:
        plot (PLOTS_DIR)
            plot and save target variable distibution to path
    '''

    