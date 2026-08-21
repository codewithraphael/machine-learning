from config import file_path
from data_loader import load_data
from data_cleaning import clean_data, investigate_time

'''
from train import train_model
from evaluate import evaluate_model
'''

import warnings; warnings.filterwarnings('ignore')


def main():
    data = load_data(file_path)
    data = clean_data(data)
    investigate_time(data)



if __name__ == '__main__':
    main()