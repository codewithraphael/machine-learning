# %% 
# ===== Importing Libraries =====
import pandas as pd
from pathlib import Path
from src.config import TARGET_COLUMN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib



import warnings
warnings.filterwarnings('ignore')


# %%
# ===== Loading Cleaned Dataset =====
def load_data(filepath):
    """Load a CSV file with fallbacks and helpful error messages.

    Tries the provided path, then the path relative to the project root
    (two levels up from this file), and finally the raw data file
    `data/raw/insurance_claim.csv` if present.
    """
    # Try the exact path first
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        pass

    # Resolve relative to project root (src/..)
    project_root = Path(__file__).resolve().parents[1]
    alt_path = project_root / filepath
    if alt_path.exists():
        return pd.read_csv(alt_path)

    # Try raw dataset as a fallback
    raw_path = project_root / 'data' / 'raw' / 'insurance_claim.csv'
    if raw_path.exists():
        return pd.read_csv(raw_path)

    # Nothing found — raise a clear error listing attempted locations
    attempted = [Path(filepath).resolve(), alt_path.resolve(), raw_path.resolve()]
    raise FileNotFoundError(
        f"Could not find '{filepath}'. Tried the following locations:\n"
        + "\n".join(str(p) for p in attempted)
        + f"\nCurrent working directory: {Path.cwd()}"
    )


# %%
# ===== Feature Selection and Data Splitting =====
def feature_selection(data):
    X = data.drop(columns=['claim_status'], axis=1)
    y = data['claim_status']

    return train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# ===== Splitting Categorical and Numerical Features For Preprocessing =====
def num_cat_split(data):
    """Return lists of numeric and categorical column names.

    Excludes the target column defined in `src.config.TARGET_COLUMN` if present.
    """
    num_col = data.select_dtypes(include=['number']).columns.tolist()
    cat_col = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure the target column isn't included in feature lists
    if TARGET_COLUMN in num_col:
        num_col.remove(TARGET_COLUMN)
    if TARGET_COLUMN in cat_col:
        cat_col.remove(TARGET_COLUMN)

    return num_col, cat_col


# %%
# ===== Data Preprocessing =====
def preprocess_data(num_cols, cat_cols):
    """Create a ColumnTransformer preprocessor given lists of column names.

    Parameters
    ----------
    num_cols : list[str]
        Numeric column names
    cat_cols : list[str]
        Categorical column names

    Returns
    -------
    ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor
# %%
