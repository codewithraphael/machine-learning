# %%

# ===== Loading Libraries =====

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go


import warnings
warnings.filterwarnings('ignore')

# %%

# ===== Loading dataset =====

insurance = pd.read_csv('../dataset/Insurance claims data.csv')
insurance.head()

# %%

# ===== Summary Statistics =====
def summary_stat(data):
    shape = data.shape
    info = data.info()
    statistics = data.describe()

    print(f"\n ===== Shape of the dataset ===== \n {shape}")
    print(f"\n ===== Information about the dataset ===== \n {info}")
    print(f"\n ===== Summary Statistics ===== \n {statistics}")

summary_stat(insurance)

# %%

# ===== Handling Outliers
def handle_outliers(data):
    q1 = data.select_dtypes(include=['number']).quantile(0.25)
    q3 = data.select_dtypes(include=['number']).quantile(0.75)
    
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(f'\n ===== Lower Bound ===== \n {lower_bound}')
    print(f'\n ===== Upper Bound ===== \n {upper_bound}')

    return lower_bound, upper_bound

lower_bound, upper_bound = handle_outliers(insurance)

# %%
# ===== Checking Outliers Counts, Bounds, Actual Min and Max Values
def check_outlier(data, lower_bound, upper_bound):
    for col in data.select_dtypes(include=['number']):
        mask = (lower_bound[col] < data[col]) | (upper_bound[col] > data[col]) 
        counts = mask.sum()
        print(f' {col}: {counts} outliers |'
              f'bounds = [{lower_bound.min():.3f} {upper_bound.max():.3f}] |'
              f'actual min = {lower_bound[col].min():.3f}, actual max = {upper_bound[col].max():.3f}'
              )

check_outlier(insurance, lower_bound=lower_bound, upper_bound=upper_bound)

# %%
# ===== Checking For Missing Values
def missing_values(data):
    values = data.isnull().sum().sort_values(ascending=False)

    print(f'\n ===== Missning Values ===== \n {values}') 

missing_values(insurance)

# %%
# ===== Visualizing the Outliers =====
figure = px.box(insurance.select_dtypes(include=['number']),
                orientation = 'h'
                )
figure.show()

# %%
# ===== Pie Chart of the Target Variable =====
claim_status_counts = insurance['claim_status'].value_counts()
labels = claim_status_counts.index
values = claim_status_counts.values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
fig.update_layout(title_text='Distribution of Claim Status')
fig.update_traces(textinfo='percent+label')
fig.show()

# %%
# ===== Correlation Heatmap =====
numeric_cols = insurance.select_dtypes(include=['number']).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols, annot=True, cmap='viridis', linewidths=0.5)
plt.title('Numerical Features Heatmap Correlation')
plt.xticks(rotation=45)
plt.show()

# %%
# ===== Pair Plot =====
plt.figure(figsize=())
sns.pairplot(insurance.select_dtypes(include=['number']))
plt.suptitle('Numerical Features Pair Plot', y=1.02)   
plt.show()