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
models = {
    "lr_model": LogisticRegression(class_weight='balanced', random_state=42),
    "rfc_model": RandomForestClassifier(class_weight='balanced', random_state=42),
    "dfc_model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "xgb_model": XGBClassifier()
}

def train_model(X_train, y_train):
    trained_model = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_model[name] = model
        

    return trained_model

