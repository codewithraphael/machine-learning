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
def load_transaction_data(file_path):

    transactions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            items = [item.strip() for item in line.split(',')]
            items = [item for item in items if item]
            items = list(set(items))

            if items:
                transactions.append(items)    





# ====================================
#  MAIN
# ====================================
def main():

    file_path = DATA_PATH
    load_transaction_data(file_path)










if __name__ == "__main__":
    main()