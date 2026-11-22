import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from config import PLOTS_DIR

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

