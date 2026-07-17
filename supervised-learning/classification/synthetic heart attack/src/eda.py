import pandas as pd
import numpy as np

import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt



def eda(data):
    shape = data.shape
    missing_value = data.isnull().sum().sort_values(ascending=False)
    info = data.info()
    summary_stat = data.describe()

    print(f'\n ===== Shape Of the Dataset ===== \n {shape}')
    print(f'\n ===== Missing Value ===== \n {missing_value}')
    print(f'{info}')
    print(f'===== Summary Statistics ===== \n {summary_stat}')


def outlier_detection(data):
    q1 = data.select_dtypes(include=['number']).quantile(0.25)
    q3 = data.select_dtypes(include=['number']).quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f'\n ===== Lower Bound ===== \n {lower_bound}')
    print(f'\n ===== Upper Bound ===== \n {upper_bound}')

    return lower_bound, upper_bound

def outlier_check(data, lower_bound, upper_bound):
    for cols in data.select_dtypes(include=['number']).columns:
        mask = (data[cols] < lower_bound[cols]) | (data[cols] > upper_bound[cols])
        count = mask.sum()
        if count > 0:
            print(f'{cols} : {count} outliers | '
              f'bounds=[{lower_bound[cols]:.2f}, {upper_bound[cols]:.2f}] | '
              f'actual_min={lower_bound[cols].min():.2f}, actual_max={upper_bound[cols].max():.2f}')


def plot_outliers(data):
        plt.figure(figsize=(22, 6))
        plt.boxplot(data.select_dtypes(include=['number']), orientation='horizontal')
        plt.title('Outlier Visualization')
        plt.savefig('../plots/outlier_visualization.png')
        plt.close()
    


def plot_pairplot(data):
    sns.pairplot(data.select_dtypes(include=['number']))
    plt.savefig('../plots/pairplot_distribution.png')
    plt.close()


def plot_correlation_matrix(data):
    
    corr_matrix = data.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, linewidths=0.5, annot=True, cmap='viridis')
    plt.savefig('../plots/correlation_matrix_plot.png')
    plt.close()

