# %%
# ===== Importing Libraries =====


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_data

# %%
# ===== Loading Dataset =====

vertebral_column = load_data('../data/vertebral_column.csv')
vertebral_column.head()

# %%
# ===== Summary Statistics =====

vertebral_column.describe()
# %%
# ===== Shape of the dataset =====
vertebral_column.shape

# %%
# ===== Infornamtion on the dataset =====
vertebral_column.info()

# %%
# ===== Checking For Missing Values =====

def missing_values(data):
    values = data.isnull().sum().sort_values(ascending=False)

    print(values)

missing_values(vertebral_column)

# %%
# ===== Checking For Outliers =====

def check_outliers(data):
    q1 = data.select_dtypes(include=['number']).quantile(0.25)
    q3 = data.select_dtypes(include=['number']).quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound

lower_bound, upper_bound = check_outliers(vertebral_column)

# %%

def outliers(data, lower_bound, upper_bound):
    for cols in data.select_dtypes(include=['number']):
        mask = (lower_bound[cols] < data[cols]) | (upper_bound[cols] > data[cols])
        counts = mask.sum()
        if counts > 0:
            print(
                f'{cols} : {counts} outliers | ',
                f'bound = [{lower_bound[cols]:.3f}, {upper_bound[cols]:.3f}] | ',
                f'actual min = {data[cols].min():.3f}, actual max = {data[cols].max():.3f}'
            )

print(outliers(vertebral_column, lower_bound, upper_bound))
# %%
# ===== Plotting Outliers =====

def plot_outliers(data):
    plt.figure(figsize=(22, 6))
    figure = px.box(data.select_dtypes(include=['number']),
                    orientation='h',
                     title= 'Outliers Visualization')
    
    figure.show()
    plt.show()

plot_outliers(vertebral_column)

# %%
# ===== Correlation HeatMap =====

def plot_correlation(data):
    corr_matrix = data.select_dtypes(include=['number']).corr()

    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix,
                annot=True, 
                linewidths=0.5, cmap='viridis')
    plt.show()

plot_correlation(vertebral_column)
# %%
# ===== Pair Plot =====

def plot_pairplot(data):
    sns.pairplot(data.select_dtypes(include=['number']))
    plt.suptitle('Vertebral Column Pair Plot')
    plt.show()

plot_pairplot(vertebral_column)