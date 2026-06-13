# %%
# ===== Loading Libraries =====

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib


import warnings
warnings.filterwarnings('ignore')




# %%
# ===== Loading dataset =====

insurance = pd.read_csv('../dataset/Insurance claims data.csv')
insurance = insurance.drop(['policy_id'], axis=1)
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
    for cols in data.select_dtypes(include=['number']).columns:
        mask = (data[cols] < lower_bound[cols]) | (data[cols] > upper_bound[cols])
        counts = mask.sum()
        if counts > 0:
            print(f' {cols} : {counts} outliers |'
                f'bounds = [{lower_bound[cols]:.3f}, {upper_bound[cols]:.3f}] |'
                f'actual min = {data[cols].min():.3f}, actual max = {data[cols].max():.3f}'
                )

check_outlier(insurance, lower_bound=lower_bound, upper_bound=upper_bound)




# %%
# ===== Checking For Missing Values
def missing_values(data):
    values = data.isnull().sum().sort_values(ascending=False)

    print(f'\n ===== Missing Values ===== \n {values}') 

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
sns.pairplot(insurance.select_dtypes(include=['number']))
plt.suptitle('Numerical Features Pair Plot')   
plt.show()





# %%
# ===== Feature Selection and Data Splitting =====
X = insurance.drop(columns=['claim_status'], axis=1)
y = insurance['claim_status']

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(insurance)




# %%
# ===== Data Type Splitting =====
def split_features(data):
    num_col = data.select_dtypes(include=['number']).drop(columns=['claim_status'], axis=1)
    cat_col = data.select_dtypes(include=['object'])
    
    return num_col, cat_col

num_col, cat_col = split_features(insurance)




# %%
# ===== Data Preprocessing =====
def preprocess_data(num_col, cat_col):
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), num_col),
            ('cat', OneHotEncoder(), cat_col)
        ]
    )

    return preprocessor

preprocessor = preprocess_data(num_col, cat_col)




# %%
# ===== Saving Preprocessed Data For Training =====
processed_data = joblib.dump(preprocessor, 'preprocessor.joblib')
# %%
