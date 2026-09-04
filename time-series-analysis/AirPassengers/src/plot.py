import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from config import PLOTS_PATH


def plot_time_series(data, target_column):

    plt.figure(figsize=(15, 5))

    plt.plot(data.index, data[target_column])
    plt.title(f'{target_column} Over time')
    plt.xlabel('Date')
    plt.ylabel(target_column)

    plt.grid(True)
    plt.savefig(PLOTS_PATH / 'passengers_distribution_over_time.png')
    plt.close()