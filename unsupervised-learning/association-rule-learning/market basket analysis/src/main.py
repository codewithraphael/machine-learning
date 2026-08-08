import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from config import DATA_PATH, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, TOP_N_ITEMS, TOP_N_RULES


# ====================================
#  LOADING TRANSACTION DATASET
# ====================================
