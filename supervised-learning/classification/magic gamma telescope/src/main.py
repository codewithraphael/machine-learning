from config import DATA_DIR, MODELS_DIR, PLOTS_DIR, EVAL_RESULT_DIR
from data_loader import load_data
from data_validation import validate_data
from data_cleaning import clean_data
from data_visualization import visualize_data


import warnings
warnings.filterwarnings('ignore')


def main():

    data = load_data(DATA_DIR)
    validate_data(data)
    clean_data(data)
    visualize_data(data)

if __name__ == '__main__':
    main()