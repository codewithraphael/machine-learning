from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import NUM_COL, CAT_COL


def preprocess_data():

    preprocessor = ColumnTransformer(
        [
            ('num', StandardScaler(), NUM_COL),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COL)
        ]
    )

    return preprocessor