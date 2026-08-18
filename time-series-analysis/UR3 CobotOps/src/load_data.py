import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import data

from config import filepath, DATA_DIR


def load_data(filepath=filepath):

    data = pd.read_excel(filepath)
    return data