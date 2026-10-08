from config import FEATURE_QUERY
from database import get_connection
from data_loader import load_sql_query
from data_validation import validate_query
from data_cleaning import clean_query
from feature_engineering import features
from visualization import visualize_query
from preprocess import *
from train import *
from evaluate import *
from predict import *

import warnings
warnings.filterwarnings('ignore')



def main():
    query = load_sql_query()
    validate_query(query)
    query = clean_query(query)
    query = features(query)
    visualize_query(query)





if __name__ == '__main__':
    main()
