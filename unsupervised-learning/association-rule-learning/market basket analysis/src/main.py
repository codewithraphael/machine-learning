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
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets['itemset_size'] = (
        frequent_itemsets['itemsets'].apply(len)
    )

    frequent_itemsets = frequent_itemsets.sort_values(
        by='support',
        ascending=False
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
        f'\n NUMBER OF FREQUENT ITEMSETS: '
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
        metric='confidence',
        min_threshold=min_confidence
    )
    
    if rules.empty:

        print('\nNo association rules found.')

        return rules

    # Calculate additional useful metrics

    rules["antecedent_length"] = (
        rules["antecedents"]
        .apply(len)
    )

    rules["consequent_length"] = (
        rules["consequents"]
        .apply(len)
    )

    # Sort by lift
    rules = rules.sort_values(
        by="lift",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("ASSOCIATION RULES")
    print("=" * 60)

    print(
        f"Number of rules: {len(rules)}"
    )

    return rules

# ============================================================
# FILTER HIGH QUALITY RULES
# ============================================================

def filter_rules(
    rules,
    min_lift=1.0,
    min_confidence=0.30
):
    """
    Filter rules using confidence and lift.
    """

    if rules.empty:
        return rules

    filtered_rules = rules[
        (rules["lift"] >= min_lift)
        &
        (rules["confidence"] >= min_confidence)
    ]

    filtered_rules = filtered_rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    print("\n" + "=" * 60)
    print("HIGH QUALITY RULES")
    print("=" * 60)

    print(
        f"Rules after filtering: "
        f"{len(filtered_rules)}"
    )

    return filtered_rules

# ============================================================
# FORMAT RULES FOR DISPLAY
# ============================================================

def format_itemset(itemset):

    return ", ".join(
        sorted(list(itemset))
    )

def prepare_rule_table(rules):

    if rules.empty:
        return pd.DataFrame()

    result = rules.copy()

    result["antecedent"] = (
        result["antecedents"]
        .apply(format_itemset)
    )

    result["consequent"] = (
        result["consequents"]
        .apply(format_itemset)
    )

    result = result[
        [
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift"
        ]
    ]

    return result

# ============================================================
# DISPLAY TOP RULES
# ============================================================

def display_top_rules(
    rules,
    top_n=20
):

    if rules.empty:

        print("\nNo rules available.")

        return

    rule_table = prepare_rule_table(rules)

    print("\n" + "=" * 60)
    print(f"TOP {top_n} ASSOCIATION RULES")
    print("=" * 60)

    display_table = rule_table.head(top_n).copy()

    display_table["support"] = (
        display_table["support"]
        .round(4)
    )

    display_table["confidence"] = (
        display_table["confidence"]
        .round(4)
    )

    display_table["lift"] = (
        display_table["lift"]
        .round(4)
    )

    print(
        display_table.to_string(
            index=False
        )
    )

# ============================================================
# PLOT TOP ASSOCIATION RULES
# ============================================================

def plot_top_rules(
    rules,
    top_n=15
):

    if rules.empty:
        return

    plot_data = rules.head(top_n).copy()

    plot_data["rule"] = (
        plot_data["antecedents"]
        .apply(format_itemset)
        +
        " → "
        +
        plot_data["consequents"]
        .apply(format_itemset)
    )

    plot_data = plot_data.sort_values(
        by="lift"
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=plot_data,
        x="lift",
        y="rule"
    )

    plt.title("Top Association Rules by Lift")

    plt.xlabel("Lift")

    plt.ylabel("Association Rule")

    plt.tight_layout()

    plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    frequent_itemsets,
    rules,
    frequency_df
):

    # Save product frequencies

    frequency_df.to_csv(
        "item_frequencies.csv",
        index=False
    )

    # Save frequent itemsets

    frequent_export = frequent_itemsets.copy()

    frequent_export["itemsets"] = (
        frequent_export["itemsets"]
        .apply(format_itemset)
    )

    frequent_export.to_csv(
        "frequent_itemsets.csv",
        index=False
    )

    # Save association rules

    if not rules.empty:

        rule_export = prepare_rule_table(rules)

        rule_export.to_csv(
            "association_rules.csv",
            index=False
        )

    print("item_frequencies.csv")
    print("frequent_itemsets.csv")
    print("association_rules.csv")


# ====================================
#  MAIN
# ====================================
def main():

    file_path = DATA_PATH
    transactions = load_transaction_data(file_path)
    eda(transactions)
    frequency_df = get_item_frequencies(transactions)
    plot_top_items(frequency_df, top_n=TOP_N_ITEMS)
    encoded_df = encode_transactions(
        transactions
    )

    frequent_itemsets = find_frequent_itemsets(
        encoded_df,
        MIN_SUPPORT
    )

    rules = generate_rules(
        frequent_itemsets,
        MIN_CONFIDENCE
    )

    rules = filter_rules(
        rules,
        MIN_LIFT,
        MIN_CONFIDENCE
    )

    display_top_rules(
        rules,
        TOP_N_RULES
    )

    plot_top_rules(
        rules,
        TOP_N_RULES
    )

    save_results(
        frequent_itemsets,
        rules,
        frequency_df
    )


if __name__ == "__main__":
    main()