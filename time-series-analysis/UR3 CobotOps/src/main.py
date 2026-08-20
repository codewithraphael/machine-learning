from config import file_path
from load_data import load_data
from preprocess import clean_data

'''
from train import train_model
from evaluate import evaluate_model
'''

import warnings; warnings.filterwarnings('ignore')


def main():
    data = load_data(file_path)
    data = clean_data(data)



if __name__ == '__main__':
    main()