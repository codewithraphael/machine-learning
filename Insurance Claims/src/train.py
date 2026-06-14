# %%
# ===== Loading Libraries =====
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


# %%
# ==== Training Model =====
'''
def train_lr_model(X_train, y_train):
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)

    return lr_model

def train_rfc_model(X_train, y_train)
    rfc_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    rfc_model.fit(X_train, y_train)

    return rfc_model

def train_dfc_model(X_train, y_train):
    dfc_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    dfc_model.fit(X_train, y_train)

    return dfc_model

def train_xgboost_model(X_train, y_train):
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)

    return xgb_model

'''


# %%
# ==== Training Model =====
models = {
    "lr_model": LogisticRegression(class_weight='balanced', random_state=42),
    "rfc_model": RandomForestClassifier(class_weight='balanced', random_state=42),
    "dfc_model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "xgb_model": XGBClassifier()
}

def train_model(X_train, y_train):
    for model in models:
        model.fit(X_train, y_train)
        

    return model

