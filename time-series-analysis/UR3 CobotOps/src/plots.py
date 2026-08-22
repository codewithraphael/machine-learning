import matplotlib.pyplot as plt

from feature_engineering import create_time_index
from config import TARGET, PLOTS_DIR


def plot_grip_loss_over_time(data):

    plt.figure(figsize=(15, 5))

    plt.plot(data['time_index'],
             data[TARGET],
             linewidth=1
    )

    plt.title(f'{TARGET} over time')
    plt.xlabel('time index')
    plt.ylabel(TARGET)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'grip_loss_over_time.png')
    plt.close()