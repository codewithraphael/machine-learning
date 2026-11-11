import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from config import PLOTS_DIR, FEATURE_COLS, TARGET_COL, COL_NAMES


def visualize_data(data, label):

    '''
    pairplot, correlation heatmap and histogram distribution plots


        sns.pairplot(
        data = data[FEATURE_COLS],
        hue = data[TARGET_COL],
        diag_kind='auto'
    )

    plt.suptitle('FEATURES PAIRPLOT')
    plt.savefig(PLOTS_DIR / 'pairplot_distribution.png')
    plt.close()
    '''


    for col in COL_NAMES[:-1]:
        plt.hist(
            data[data['class'] == 1][label], label='gamma', color=' blue', alpha=0.7, density=True 
        )
        plt.hist(
            data[data['class'] == 0][label], label='hadron', color=' red', alpha=0.7, density=True 
        )
        plt.title(label)
        plt.savefig(PLOTS_DIR / ' {data}probability_distribution.png')






    plt.figure(figsize=(20, 10))
    corr_matrix = data[FEATURE_COLS].corr()   
    sns.heatmap(
        corr_matrix,
        annot=True,
        linewidths=0.5,
        cmap='viridis',
        fmt='.2f'
    )

    plt.suptitle('CORRELATION HEATMAP')
    plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()