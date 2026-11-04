import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from config import PLOTS_DIR


def visualize_data(data):
    """
    Visualize the distribution of features in the dataset.

    Args:
        data (pd.DataFrame): The dataset to visualize.
    """

    pairplot = sns.pairplot(data, hue='outcome', diag_kind='kde')
    pairplot.figure.suptitle('Pairplot of Features', y=1.02)
    pairplot.savefig(PLOTS_DIR / 'pairplot.png')
    plt.close(pairplot.figure)


    plt.figure(figsize=(20, 10))
    correlation_matrix = data.corr()
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    heatmap.set_title('Correlation Heatmap')
    plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()