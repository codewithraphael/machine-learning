from config import (
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    NUM_COL,
    CAT_COL,
    TARGET_COL,
    RANDOM_STATE
)

from data_loader import load_data
from data_validation import (
    validate_data,
    clean_data
)

from eda import eda
from preprocessing import preprocess_data

from train import (
    split_features, 
    train_models
)



import warnings; warnings.filterwarnings('ignore')



def main():

    data = load_data(DATA_DIR)
    validate_data(data)
    data = clean_data(data)
    eda(data)

    preprocessor = preprocess_data()
    x_train, x_test, y_train, y_test = split_features(data)
    trained_models = train_models(x_train, y_train, preprocessor)

if __name__ == '__main__':
    main()