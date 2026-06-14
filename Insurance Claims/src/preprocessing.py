# %% 
# ===== Importing Libraries =====
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib



import warnings
warnings.filterwarnings('ignore')


# %%
# ===== Loading Cleaned Dataset =====
clean_insurance = pd.read_csv('../data/processed/clean_insurance.csv')
clean_insurance.head()


# %%
# ===== Feature Selection and Data Splitting =====
def feature_selection(data):
    X = data.drop(columns=['claim_status'], axis=1)
    y = data['claim_status']

    return train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# ===== Splitting Categorical and Numerical Features For Preprocessing =====
def num_cat_split(data):
    num_col = data.select_dtypes(include=['number'])
    cat_col = data.select_dtypes(include=['object'])

    return num_col, cat_col
num_col, cat_col = num_cat_split(clean_insurance)

# %%
# ===== Data Preprocessing =====
def preprocess_data(num_col, cat_col):
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', StandardScaler(), num_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
        ]
    )

    return preprocessor

preprocessor = preprocess_data(num_col, cat_col)

# %%
# ===== Saving Preprocessed Data For Training =====
processed_data = joblib.dump(preprocessor, '../models/preprocessor.joblib')