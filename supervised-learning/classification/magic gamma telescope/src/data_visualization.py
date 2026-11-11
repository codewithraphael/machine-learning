import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURE_COLS, PLOTS_DIR, TARGET_COL


sns.set_theme()


def visualize_data(data):
    """Save class-wise feature distributions, a feature pairplot, and correlations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    classes = ((1, "gamma", "blue"), (0, "hadron", "red"))
    for feature in FEATURE_COLS:
        plt.figure()
        for class_value, class_name, color in classes:
            values = data.loc[data[TARGET_COL] == class_value, feature]
            plt.hist(
                values,
                bins=30,
                label=class_name,
                color=color,
                alpha=0.7,
                density=True,
            )

        plt.title(f"{feature} probability distribution")
        plt.xlabel(feature)
        plt.ylabel("Probability density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{feature}_probability_distribution.png")
        plt.close()

    pairplot = sns.pairplot(
        data=data[FEATURE_COLS + [TARGET_COL]],
        hue=TARGET_COL,
        diag_kind="auto",
    )
    pairplot.fig.suptitle("FEATURES PAIRPLOT", y=1.02)
    pairplot.savefig(PLOTS_DIR / "pairplot_distribution.png")
    plt.close(pairplot.fig)

    plt.figure(figsize=(20, 10))
    corr_matrix = data[FEATURE_COLS].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        linewidths=0.5,
        cmap="viridis",
        fmt=".2f",
    )
    plt.title("CORRELATION HEATMAP")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()
