import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from config import PLOTS_DIR


def visualize_query(query):

    fig, axes = plt.subplots(2, 5, figsize=(30, 10))

    sns.countplot(x = 'city', data = query, color='skyblue', ax = axes[0, 0])
    axes[0, 0].set_title('City')

    sns.countplot(x = 'state', data = query, color='red', ax = axes[0, 1])
    axes[0, 1].set_title('State')

    sns.countplot(x='is_active', data = query, color = 'green', ax = axes[0, 2])
    axes[0, 2].set_title('Order Status')

    sns.countplot(x = 'total_orders', data = query, color = 'purple', ax = axes[0, 3])
    axes[0, 3].set_title('Total Orders')

    sns.countplot(x = 'unique_products', data = query, color = 'pink', ax = axes[0, 4])
    axes[0, 4].set_title('Unique Products')

    sns.histplot(x = 'total_spent', data = query, color = 'brown', ax = axes[1, 0], kde = True)
    axes[1, 0].set_title('Total Spent')

    sns.histplot(x = 'average_item_value', data = query, color = 'violet', ax = axes[1, 1], kde = True)
    axes[1, 1].set_title('Average Item Value')

    sns.histplot(x = 'average_order_value', data = query, color = 'magenta', ax = axes[1, 2], kde = True)
    axes[1, 2].set_title('Average Order Value')

    sns.countplot(x = 'products_per_order', data = query, color = 'grey', ax = axes[1, 3])
    axes[1, 3].set_title('Products Per Order')

    num_col = query.select_dtypes(include=[np.number]).corr()
    sns.heatmap(num_col, ax=axes[1, 4], annot=False, cmap='viridis')
    axes[1, 4].set_title('Heatmap Correlation')


    for ax in axes.flat:
        ax.tick_params(axis='x', labelrotation=45)


    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'data_visualization.png')
    plt.close()