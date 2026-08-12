import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from pathlib import Path

from config import DATA_PATH, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, TOP_N_ITEMS, TOP_N_RULES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / 'plots'

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================================
#  LOADING TRANSACTION DATASET
# ====================================
def load_transaction_data(file_path):

    print('='*80)
    print(f"MARKET BASKET ANALYSIS")
    print('='*80)

    transactions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            items = [item.strip().lower() for item in line.split(',')]
            items = [item for item in items if item]
            items = list(set(items))

            if items:
                transactions.append(items)    

    return transactions


# ====================================
#  EXPLANATORY DATA ANALYSIS
# ====================================
def eda(transactions):

    transaction_sizes = [len(transaction) for transaction in transactions]
    total_items = sum(transaction_sizes)
    average_items = np.mean(transaction_sizes)

    print('='*60)
    print(f"EXPLANATORY DATA ANALYSIS")
    print('='*60)
    print(transactions[:10])
    print(f"\n TOTAL NUMBER OF TRANSACTIONS: {len(transactions)}")
    print(f'\n TOTAL ITEMS PURCHASED: {total_items}')
    print(f'\n AVERAGE ITEMS PER TRANSACTION: {average_items:.2f}')
    print(f'\n MINIMUM ITEMS IN TRANSACTION: {min(transaction_sizes)}')
    print(f'\n MAXIMUM ITEMS IN TRANSACTION: {max(transaction_sizes)}')



# ====================================
#  COUNT INDIVIDUAL ITEMS
# ====================================
def get_item_frequencies(transactions):

    item_counts = {}

    for transaction in transactions:
        for item in transaction:

            if item not in item_counts:
                item_counts[item] = 0

            item_counts[item] += 1


    frequency_df = pd.DataFrame(
        list(item_counts.items()),
        columns=['item', 'frequency']
    )

    frequency_df = frequency_df.sort_values(
        by='frequency',
        ascending=False
    )

    frequency_df = frequency_df.reset_index(drop=True)

    print(f'\n ===== TOP MOST POPULAR SELLING PRODUCTS =====')
    print(frequency_df.head(10))

    return frequency_df



# ====================================
#  VISUALIZING TOP PRODUCTS
# ====================================

def plot_top_items(frequency_df, top_n=15):

    top_items = frequency_df.head(top_n)

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top_items,
        x="frequency",
        y="item"
    )

    plt.title("Top Purchased Products")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'top_purchased_products.png')
    plt.close()


# ====================================
#  TRANSACTION ENCODING
# ====================================
def encode_transactions(transactions):

    te = TransactionEncoder()
    encoded_array = te.fit(transactions).transform(transactions)

    encoded_df = pd.DataFrame(encoded_array, columns=te.columns_)

    return encoded_df


# =======================================================
#  FIND FREQUENT ITEMSETS USING APRIORI ALGORITHM
# =======================================================
def find_frequent_itemsets(encoded_df, min_support=MIN_SUPPORT):

    frequent_itemsets = apriori(
        encoded_df,
        min_support,
        use_colnames=True
    )

    frequent_itemsets['itemset_size'] = (
        frequent_itemsets['itemsets'].apply(len)
    )

    frequent_itemsets = frequent_itemsets.sort_values(
        by='support',
        ascending='False'
    )

    print('\n' + '=' * 60)
    print('FREQUENT ITEMSETS')
    print('=' * 60)

    print(
        frequent_itemsets.head(20).to_string(
            index=False
        )
    )

    print(
        f'\n NUMBR OF FREQUENT ITEMSETS: '
        f'{len(frequent_itemsets)}'
    )

    return frequent_itemsets


# ====================================
#  GENERATING ASSOCIATION RULES
# ====================================
def generate_rules(frequent_itemsets, min_confidence=MIN_CONFIDENCE):

    if frequent_itemsets.empty:

        print('\nNo frequent itemsets found.')
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metrics='confidence',
        min_threshold=min_confidence
    )
    
    if rules.empty:

        print('\nNo association rules found.')

        return rules

    rules['antecendant_length'] = 











# ====================================
#  MAIN
# ====================================
def main():

    file_path = DATA_PATH
    transactions = load_transaction_data(file_path)
    eda(transactions)
    frequency_df = get_item_frequencies(transactions)
    plot_top_items(frequency_df, top_n=20)


if __name__ == "__main__":
    main()